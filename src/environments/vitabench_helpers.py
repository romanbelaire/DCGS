"""Helper functions for VitaBench environment integration."""

import json
import sys
import copy
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from ..training.episode_state import EpisodeState
from .vitabench_env import VitaBenchEnvironment
from ..utils.llm_utils import (
    is_local_model,
    get_vitabench_model_instance,
    messages_to_prompt,
    batch_generate
)

# Add vitabench directory to Python path if not already present
_project_root = Path(__file__).parent.parent.parent
_vitabench_path = _project_root / 'vitabench' / 'src'
if _vitabench_path.exists() and str(_vitabench_path) not in sys.path:
    sys.path.insert(0, str(_vitabench_path))


# Request collectors for monkey-patched generate() (single-step / tool-call path; main batch uses vitabench_env.batch_step_episodes)
_user_requests: List[Dict] = []
_evaluator_requests: List[Dict] = []
_request_counter = 0
_vitabench_user_is_local = False

# Legacy: set when local user model is used (create_episode_state); batch_step_episodes gets model from env
_vitabench_user_model = None
_vitabench_user_tokenizer = None


def _prepare_user_sim_messages(env, agent_message) -> List:
    """
    Build the messages list that would be passed to the user-sim LLM, without calling generate.
    Does not mutate env.orchestrator state.
    """
    from copy import deepcopy
    from vita.data_model.message import AssistantMessage

    state = deepcopy(env.orchestrator.user_state)
    if hasattr(agent_message, "tool_messages"):
        state.messages.extend(agent_message.tool_messages)
    else:
        state.messages.append(agent_message)
    flipped = state.flip_roles()
    limited = flipped[-2:] if len(flipped) > 2 else flipped
    cleaned = []
    for msg in limited:
        if isinstance(msg, AssistantMessage) and getattr(msg, "tool_calls", None):
            msg_copy = deepcopy(msg)
            msg_copy.tool_calls = None
            cleaned.append(msg_copy)
        else:
            cleaned.append(msg)
    return state.system_messages + cleaned


def _is_tool_call_action(action: str) -> bool:
    """True if action is a JSON tool call."""
    try:
        data = json.loads(action)
        return isinstance(data, dict) and "name" in data and "arguments" in data
    except (json.JSONDecodeError, TypeError):
        return False


def batch_vitabench_user_responses(episodes: List, config) -> None:
    """
    Batch user-sim steps for VitaBench (local models). Delegates to vitabench_env.batch_step_episodes
    so processing logic lives in the env; no async/barriers.
    """
    from .vitabench_env import batch_step_episodes
    batch_step_episodes(episodes, config)


def load_vitabench_task(domain: str, task_id: str, language: str = "english"):
    """Load a VitaBench task."""
    from vita.run import load_tasks
    
    tasks = load_tasks(domain, language=language)
    task = next((t for t in tasks if t.id == task_id), None)
    if task is None:
        raise ValueError(f"Task {task_id} not found in domain {domain}")
    return task


def create_vitabench_episode_state(
    task_id: str,
    domain: str,
    config,
    debug: bool = False,
    language: str = "english",
    _preloaded: Optional[Tuple] = None,
) -> EpisodeState:
    """Create EpisodeState from VitaBench task. When _preloaded=(user_model, user_tokenizer, evaluator_model, evaluator_tokenizer), skip model load and monkey-patch (for batch init)."""
    from vita.environment.environment import get_cross_environment
    from vita.user.user_simulator import UserSimulator
    from vita.orchestrator.orchestrator import Orchestrator
    import json
    
    # Load task
    task = load_vitabench_task(domain, task_id, language=language)
    
    # Create VitaBench environment
    # Some task loaders/configurations may serialize the environment as JSON.
    # Normalize here so VitaBench env constructors receive a dict.
    if isinstance(task.environment, str):
        task.environment = json.loads(task.environment)

    if "," in domain:
        vitabench_env = get_cross_environment(domain, task.environment, language)
    else:
        from vita.registry import registry
        environment_constructor = registry.get_env_constructor(domain)
        vitabench_env = environment_constructor(task.environment, language)
    
    # Configure user, agent, and evaluator models (local or API)
    import os
    import copy
    from vita.config import DEFAULT_LLM_USER, DEFAULT_LLM_EVALUATOR, models
    from vita.user.user_simulator import UserSimulator
    from vita.data_model.message import AssistantMessage
    
    # Get device and precision settings
    device = getattr(config, 'device', 'cuda')
    use_bf16 = getattr(config, 'use_bf16', True)
    
    # Configure user model
    llm_user = getattr(config, 'vitabench_llm_user', None)
    if llm_user is None:
        llm_user = "gpt-4o-mini"
    
    user_model = None
    user_tokenizer = None
    user_is_local = is_local_model(llm_user)
    evaluator_model = None
    evaluator_tokenizer = None
    if _preloaded is not None:
        user_model, user_tokenizer, evaluator_model, evaluator_tokenizer = _preloaded
    if user_is_local and user_model is None:
        user_model, user_tokenizer = get_vitabench_model_instance(
            model_name=llm_user,
            device=device,
            use_bf16=use_bf16
        )
        global _vitabench_user_model, _vitabench_user_tokenizer
        _vitabench_user_model = user_model
        _vitabench_user_tokenizer = user_tokenizer
        print(f"[INFO] Loaded local user model: {llm_user}")
    if user_is_local:
        if llm_user not in models:
            models[llm_user] = {"max_tokens": 256}
        llm_args_user = models[llm_user]
    else:
        api_key = os.getenv("OPENAI_API_KEY") or getattr(config, 'openai_api_key', None)
        if api_key and isinstance(api_key, str) and api_key.strip():
            api_key = api_key.strip()
        else:
            api_key = None
        if hasattr(config, 'vitabench_llm_args_user') and config.vitabench_llm_args_user:
            llm_args_user = copy.deepcopy(config.vitabench_llm_args_user)
        elif llm_user in models:
            llm_args_user = copy.deepcopy(models[llm_user])
        else:
            llm_args_user = {}
        if api_key:
            openai_endpoint = 'https://api.openai.com/v1/chat/completions'
            if llm_user not in models:
                raise ValueError(f"Model {llm_user} not found in models.yaml")
            models[llm_user]['base_url'] = openai_endpoint
            if 'headers' not in models[llm_user]:
                models[llm_user]['headers'] = {}
            models[llm_user]['headers']['Authorization'] = f"Bearer {api_key}"
            models[llm_user]['headers']['Content-Type'] = "application/json"
            print(f"[INFO] Configured VitaBench UserSimulator with OpenAI API (model: {llm_user})")
        else:
            current_base_url = llm_args_user.get('base_url', '')
            if current_base_url in ['<base_url>', None, '']:
                raise ValueError(
                    "VitaBench UserSimulator requires API configuration. "
                    "Set OPENAI_API_KEY or use a local model."
                )
    
    llm_evaluator = getattr(config, 'vitabench_llm_evaluator', None)
    if llm_evaluator is None:
        llm_evaluator = DEFAULT_LLM_EVALUATOR
    evaluator_is_local = is_local_model(llm_evaluator)
    if evaluator_is_local and evaluator_model is None:
        if llm_evaluator == llm_user and user_is_local:
            evaluator_model = user_model
            evaluator_tokenizer = user_tokenizer
        else:
            evaluator_model, evaluator_tokenizer = get_vitabench_model_instance(
                model_name=llm_evaluator,
                device=device,
                use_bf16=use_bf16
            )
            print(f"Loaded local evaluator model: {llm_evaluator}")
    elif not evaluator_is_local:
        api_key = os.getenv("OPENAI_API_KEY") or getattr(config, 'openai_api_key', None)
        if api_key and isinstance(api_key, str) and api_key.strip():
            api_key = api_key.strip()
        else:
            api_key = None
        
        if hasattr(config, 'vitabench_llm_args_evaluator') and config.vitabench_llm_args_evaluator:
            llm_args_evaluator = copy.deepcopy(config.vitabench_llm_args_evaluator)
        elif llm_evaluator in models:
            llm_args_evaluator = copy.deepcopy(models[llm_evaluator])
        else:
            llm_args_evaluator = {}
        
        if api_key:
            openai_endpoint = 'https://api.openai.com/v1/chat/completions'
            if llm_evaluator not in models:
                raise ValueError(f"Model {llm_evaluator} not found in models.yaml")
            
            models[llm_evaluator]['base_url'] = openai_endpoint
            if 'headers' not in models[llm_evaluator]:
                models[llm_evaluator]['headers'] = {}
            models[llm_evaluator]['headers']['Authorization'] = f"Bearer {api_key}"
            models[llm_evaluator]['headers']['Content-Type'] = "application/json"
            print(f"[INFO] Configured VitaBench evaluator with OpenAI API (model: {llm_evaluator})")
    
    # Create generate wrapper for local models with simple batching (only on first env creation; skip when _preloaded)
    original_generate = None
    if _preloaded is None and (user_is_local or evaluator_is_local):
        from vita.utils import llm_utils as vita_llm_utils
        from vita.data_model.message import AssistantMessage
        original_generate = vita_llm_utils.generate
        
        # Initialize simple request collectors
        global _user_requests, _evaluator_requests, _request_counter, _vitabench_user_is_local
        _user_requests = []
        _evaluator_requests = []
        _request_counter = 0
        _vitabench_user_is_local = user_is_local
        
        minibatch_size = getattr(config, 'vitabench_minibatch_size', 8)
        
        # Capture variables for closure
        _user_is_local = user_is_local
        _evaluator_is_local = evaluator_is_local
        _llm_user = llm_user
        _llm_evaluator = llm_evaluator
        _user_model = user_model
        _user_tokenizer = user_tokenizer
        _evaluator_model = evaluator_model
        _evaluator_tokenizer = evaluator_tokenizer
        
        def process_request_batch(requests, local_model, local_tokenizer):
            """Process a batch of requests synchronously - single tensor operation."""
            if not requests:
                return []
            
            # Convert all messages to prompts
            prompts = [messages_to_prompt(req['messages']) for req in requests]
            
            # Get generation parameters (use first request's params, should be same)
            max_new_tokens = requests[0]['kwargs'].get('max_tokens', 256)
            temperature = requests[0]['kwargs'].get('temperature', 0.7)
            do_sample = temperature > 0.0
            
            # Batch generate all prompts - single tensor operation
            generated_texts = batch_generate(
                model=local_model,
                tokenizer=local_tokenizer,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                chunk_size=minibatch_size
            )
            
            # Create AssistantMessages for results
            results = []
            for generated_text in generated_texts:
                result = AssistantMessage(
                    role="assistant",
                    content=generated_text,
                    cost=0.0,
                    usage={"prompt_tokens": 0, "completion_tokens": 0}
                )
                results.append(result)
            
            return results
        
        def local_model_generate(
            model: str,
            messages,
            tools=None,
            tool_choice=None,
            enable_think=False,
            **kwargs
        ):
            global _user_requests, _evaluator_requests
            # Check if this is a local model we're managing
            if _user_is_local and model == _llm_user:
                local_model = _user_model
                local_tokenizer = _user_tokenizer
                requests_list = _user_requests
            elif _evaluator_is_local and model == _llm_evaluator:
                local_model = _evaluator_model
                local_tokenizer = _evaluator_tokenizer
                requests_list = _evaluator_requests
            else:
                # Not a local model we manage, use original generate
                if is_local_model(model):
                    raise ValueError(
                        f"Local model {model} is not configured. "
                        f"Expected user model: {_llm_user if _user_is_local else 'N/A'}, "
                        f"evaluator model: {_llm_evaluator if _evaluator_is_local else 'N/A'}"
                    )
                return original_generate(model, messages, tools, tool_choice, enable_think, **kwargs)
            
            # Collect prompt/state (we're collecting prompts/states, not generate() calls)
            request_idx = len(requests_list)
            requests_list.append({
                'messages': messages,
                'tools': tools,
                'tool_choice': tool_choice,
                'enable_think': enable_think,
                'kwargs': kwargs
            })
            
            # Process immediately. A barrier (wait for N user-sim calls) was removed because
            # fewer than N episodes may call generate() (e.g. early return from process_turn_for_episode),
            # which caused permanent deadlock with local models.
            batch_requests = requests_list[:]
            requests_list.clear()
            results = process_request_batch(batch_requests, local_model, local_tokenizer)
            return results[request_idx]
        
        def flush_batched_generates():
            """Process any remaining collected requests (from single-step/tool path). Main batch uses batch_step_episodes."""
            global _user_requests, _evaluator_requests
            if _user_requests and _user_is_local:
                results = process_request_batch(_user_requests, _user_model, _user_tokenizer)
                _user_requests.clear()
            if _evaluator_requests and _evaluator_is_local:
                results = process_request_batch(_evaluator_requests, _evaluator_model, _evaluator_tokenizer)
                _evaluator_requests.clear()
        
        # Store flush function for later use (accessible from main.py)
        import sys
        sys.modules[__name__].flush_batched_vitabench_generates = flush_batched_generates
        
        # Monkey-patch vitabench's generate function in the module
        vita_llm_utils.generate = local_model_generate
        
        # Also patch in modules that import generate directly
        # This is needed because Python's "from X import Y" creates a local reference
        import sys
        import importlib
        
        # Ensure modules are imported, then patch their local generate references
        modules_to_patch = [
            'vita.user.user_simulator',
            'vita.agent.llm_agent',
            'vita.evaluator.evaluator_traj'
        ]
        
        for module_name in modules_to_patch:
            if module_name not in sys.modules:
                # Import the module to ensure it's loaded
                importlib.import_module(module_name)
            # Patch the local generate reference in the module
            if hasattr(sys.modules[module_name], 'generate'):
                sys.modules[module_name].generate = local_model_generate
                #print(f"[INFO] Patched generate() in {module_name}")
            else:
                print(f"[WARNING] Module {module_name} does not have 'generate' attribute")
        
        # Also create a batched wrapper for TrajectoryEvaluator.calculate_reward
        if evaluator_is_local:
            from vita.evaluator import evaluator_traj
            from vita.utils import evaluator_extracter, get_weekday
            from vita.prompts import get_prompts
            from vita.data_model.message import SystemMessage, UserMessage
            from vita.data_model.simulation import RewardInfo, RewardType
            original_calculate_reward = evaluator_traj.TrajectoryEvaluator.calculate_reward
            
            def batched_calculate_reward(
                cls,
                task,
                full_trajectory,
                final_state,
                window_size=10,
                overlap=2,
                llm_evaluator=None,
                llm_args_evaluator=None,
                language=None,
            ):
                # Only batch if using local evaluator model
                if llm_evaluator == _llm_evaluator and _evaluator_is_local:
                    # Collect all windows first
                    evaluation_criteria = task.evaluation_criteria
                    if evaluation_criteria is None:
                        return original_calculate_reward(
                            cls, task, full_trajectory, final_state,
                            window_size, overlap, llm_evaluator, llm_args_evaluator, language
                        )
                    
                    if not evaluation_criteria.expected_states and not evaluation_criteria.overall_rubrics:
                        return original_calculate_reward(
                            cls, task, full_trajectory, final_state,
                            window_size, overlap, llm_evaluator, llm_args_evaluator, language
                        )
                    
                    env_info = {
                        "system_time": "",
                        "database": []
                    }
                    if hasattr(task, 'environment') and task.environment:
                        time_str = task.environment.get("time", "")
                        if time_str:
                            weekday = get_weekday(time_str, language)
                            env_info["system_time"] = f"{time_str} {weekday or ''}"
                    
                    current_rubric_states = evaluator_traj.TrajectoryEvaluator._initialize_rubric_states(evaluation_criteria)
                    windows = evaluator_traj.TrajectoryEvaluator._create_sliding_windows(full_trajectory, window_size, overlap)
                    
                    # Process windows in batches for efficiency
                    # Note: We still need to update rubric states sequentially, so we batch
                    # the forward passes but process results in order
                    step = window_size - overlap
                    prompts_obj = get_prompts(language)
                    window_evaluations = []
                    
                    # Prepare prompts for all windows at once; minibatching is handled
                    # inside batch_generate via chunk_size/vitabench_minibatch_size.
                    prompts_batch = []
                    window_metadata = []
                    
                    for global_idx, window in enumerate(windows):
                            window_start_idx = global_idx * step
                            window_content = evaluator_traj.TrajectoryEvaluator._format_window_content(window, window_start_idx)
                            current_rubrics_str = evaluator_traj.TrajectoryEvaluator._format_current_rubrics(current_rubric_states)
                            
                            system_prompt = prompts_obj.sliding_window_eval_template.format(
                                env_info=env_info,
                                user_instruction=task.instructions,
                                window_idx=global_idx+1,
                                total_windows=len(windows)
                            )
                            user_prompt = f"""
# Input
<window_content>
{window_content}
</window_content>

<current_rubrics>
{current_rubrics_str}
</current_rubrics>
"""
                            full_prompt = messages_to_prompt([
                                SystemMessage(role="system", content=system_prompt),
                                UserMessage(role="user", content=user_prompt),
                            ])
                            prompts_batch.append(full_prompt)
                            window_metadata.append({
                                'window': window,
                                'window_idx': global_idx,
                                'window_start_idx': window_start_idx,
                                'system_prompt': system_prompt,
                                'user_prompt': user_prompt
                            })
                    
                    # Batch generate all prompts with minibatching handled by chunk_size
                    max_new_tokens = llm_args_evaluator.get('max_tokens', 256) if llm_args_evaluator else 256
                    temperature = llm_args_evaluator.get('temperature', 0.7) if llm_args_evaluator else 0.7
                    do_sample = temperature > 0.0
                    
                    generated_texts = batch_generate(
                        model=_evaluator_model,
                        tokenizer=_evaluator_tokenizer,
                        prompts=prompts_batch,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample,
                        chunk_size=getattr(config, "vitabench_minibatch_size", None)
                    )
                    
                    # Process results and update rubric states sequentially
                    for generated_text, metadata in zip(generated_texts, window_metadata):
                        # Parse result and update states
                        result_data = evaluator_extracter(generated_text)
                        # Local models may return a JSON object (dict) instead of a list; iterating yields keys (str).
                        if isinstance(result_data, dict):
                            result_data = [dict(rubric_idx=k, **v) for k, v in result_data.items()]
                        if result_data:
                            for result in result_data:
                                rubric_idx = result.get("rubric_idx")
                                if rubric_idx and rubric_idx in current_rubric_states:
                                    current_rubric_states[rubric_idx]["justification"] = result.get("justification", "No justification provided")
                                    current_rubric_states[rubric_idx]["meetExpectation"] = result.get("meetExpectation", current_rubric_states[rubric_idx]["meetExpectation"])
                        
                        window_evaluations.append({
                            "window_idx": metadata['window_idx'] + 1,
                            "system_prompt": metadata['system_prompt'],
                            "user_prompt": metadata['user_prompt'],
                            "assistant_message_content": generated_text,
                            "assistent_message_usage": {"prompt_tokens": 0, "completion_tokens": 0}
                        })
                    
                    # Convert final states to checks
                    final_nl_rubric_checks = evaluator_traj.TrajectoryEvaluator._convert_states_to_checks(current_rubric_states)
                    all_expectations_met = all(result.met for result in final_nl_rubric_checks) and len(final_nl_rubric_checks) > 0
                    rubric_score = sum(1.0 if result.met else 0.0 for result in final_nl_rubric_checks) / len(final_nl_rubric_checks) if final_nl_rubric_checks else 0.0
                    reward = 1.0 if all_expectations_met else 0.0
                    
                    return RewardInfo(
                        reward=reward,
                        nl_rubrics=final_nl_rubric_checks,
                        reward_breakdown={RewardType.NL_ASSERTION: rubric_score},
                        info={"evaluation_method": "sliding_window_batched", "num_windows": len(windows), "window_size": window_size},
                        window_evaluations=window_evaluations
                    )
                else:
                    # Not local evaluator, use original
                    return original_calculate_reward(
                        cls, task, full_trajectory, final_state,
                        window_size, overlap, llm_evaluator, llm_args_evaluator, language
                    )
            
            # Monkey-patch the class method
            evaluator_traj.TrajectoryEvaluator.calculate_reward = classmethod(batched_calculate_reward)
    
    # Configure user simulator
    if user_is_local:
        # For local models, we still need to register in models dict for vitabench
        # But we'll use our wrapper
        if llm_user not in models:
            models[llm_user] = {"max_tokens": 256}
        llm_args_user = models[llm_user]
    else:
        if hasattr(config, 'vitabench_llm_args_user') and config.vitabench_llm_args_user:
            llm_args_user = copy.deepcopy(config.vitabench_llm_args_user)
        elif llm_user in models:
            llm_args_user = copy.deepcopy(models[llm_user])
        else:
            llm_args_user = {}
    
    user_sim = UserSimulator(
        tools=vitabench_env.get_tools(),
        persona=str(task.user_scenario.user_profile),
        instructions=str(task.instructions),
        llm=llm_user,
        llm_args=llm_args_user,
        language=language
    )
    
    # Placeholder agent for the orchestrator; actions are injected via env.step(action) by the driver (batched).
    from .vitabench_env import PlaceholderAgent
    placeholder_agent = PlaceholderAgent(
        tools=vitabench_env.get_tools(),
        domain_policy=vitabench_env.get_policy(),
    )
    
    # Create orchestrator
    orchestrator = Orchestrator(
        domain=domain,
        agent=placeholder_agent,
        user=user_sim,
        environment=vitabench_env,
        task=task,
        max_steps=config.max_turns if hasattr(config, 'max_turns') else 300,
        max_errors=10,
        language=language
    )
    
    # Configure evaluator args
    if evaluator_is_local:
        if llm_evaluator not in models:
            models[llm_evaluator] = {"max_tokens": 256}
        llm_args_evaluator = models[llm_evaluator]
    else:
        if hasattr(config, 'vitabench_llm_args_evaluator') and config.vitabench_llm_args_evaluator:
            llm_args_evaluator = copy.deepcopy(config.vitabench_llm_args_evaluator)
        elif llm_evaluator in models:
            llm_args_evaluator = copy.deepcopy(models[llm_evaluator])
        else:
            llm_args_evaluator = {}
    
    # Create adapter (local models and batching are configured above).
    # _slot_init=True for batch slots (episodes 2..N) so we only print "initialized" once.
    env = VitaBenchEnvironment(
        domain_name=domain,
        task=task,
        vitabench_env=vitabench_env,
        orchestrator=orchestrator,
        max_steps=config.max_turns if hasattr(config, 'max_turns') else 300,
        debug=debug,
        llm_evaluator=llm_evaluator,
        llm_args_evaluator=llm_args_evaluator,
        evaluator_model=evaluator_model,
        evaluator_tokenizer=evaluator_tokenizer,
        user_model=user_model,
        user_tokenizer=user_tokenizer,
        _slot_init=_preloaded is not None,
    )
    
    # Get ground truth goal from task
    ground_truth_goal = task.user_scenario.goal_description if hasattr(task.user_scenario, 'goal_description') else str(task.instructions)
    
    # Create EpisodeState
    initial_obs, initial_info = env.reset()
    
    episode = EpisodeState(
        dialogue_idx=0,
        dialogue_id=task_id,
        dialogue_data=task,
        ground_truth_goal=ground_truth_goal,
        env=env,
        initial_observation=initial_obs,
        initial_env_info=initial_info
    )
    episode.initialize()
    return episode


def vitabench_user_is_local(config) -> bool:
    """True when VitaBench is configured to use a local user model (batch init is used)."""
    llm_user = getattr(config, "vitabench_llm_user", None) or "gpt-4o-mini"
    return is_local_model(llm_user)


def _vitabench_sliding_window_prompts(
    task,
    full_trajectory: List,
    window_size: int = 2,
    overlap: int = 1,
    language: Optional[str] = None,
) -> Tuple[List[str], int]:
    """Build sliding-window eval prompts for one episode. Returns (prompts_list, num_windows)."""
    from vita.evaluator import evaluator_traj
    from vita.utils import get_weekday
    from vita.prompts import get_prompts
    from vita.data_model.message import SystemMessage, UserMessage

    evaluation_criteria = task.evaluation_criteria
    if evaluation_criteria is None:
        return [], 0
    if not evaluation_criteria.expected_states and not evaluation_criteria.overall_rubrics:
        return [], 0

    env_info = {"system_time": "", "database": []}
    if hasattr(task, "environment") and task.environment:
        time_str = task.environment.get("time", "")
        if time_str:
            weekday = get_weekday(time_str, language)
            env_info["system_time"] = f"{time_str} {weekday or ''}"

    current_rubric_states = evaluator_traj.TrajectoryEvaluator._initialize_rubric_states(evaluation_criteria)
    windows = evaluator_traj.TrajectoryEvaluator._create_sliding_windows(full_trajectory, window_size, overlap)
    step = window_size - overlap
    prompts_obj = get_prompts(language)
    prompts_batch = []

    for global_idx, window in enumerate(windows):
        window_start_idx = global_idx * step
        window_content = evaluator_traj.TrajectoryEvaluator._format_window_content(window, window_start_idx)
        current_rubrics_str = evaluator_traj.TrajectoryEvaluator._format_current_rubrics(current_rubric_states)
        system_prompt = prompts_obj.sliding_window_eval_template.format(
            env_info=env_info,
            user_instruction=task.instructions,
            window_idx=global_idx + 1,
            total_windows=len(windows),
        )
        user_prompt = f"""
# Input
<window_content>
{window_content}
</window_content>

<current_rubrics>
{current_rubrics_str}
</current_rubrics>
"""
        full_prompt = messages_to_prompt([
            SystemMessage(role="system", content=system_prompt),
            UserMessage(role="user", content=user_prompt),
        ])
        prompts_batch.append(full_prompt)

    return prompts_batch, len(windows)


def _vitabench_reward_from_window_results(evaluation_criteria, generated_texts: List[str]):
    """Compute RewardInfo from sliding-window evaluator outputs (one episode)."""
    from vita.evaluator import evaluator_traj
    from vita.utils import evaluator_extracter
    from vita.data_model.simulation import RewardInfo, RewardType

    current_rubric_states = evaluator_traj.TrajectoryEvaluator._initialize_rubric_states(evaluation_criteria)
    for generated_text in generated_texts:
        result_data = evaluator_extracter(generated_text)
        if isinstance(result_data, dict):
            result_data = [dict(rubric_idx=k, **v) for k, v in result_data.items()]
        if result_data:
            for result in result_data:
                rubric_idx = result.get("rubric_idx")
                if rubric_idx and rubric_idx in current_rubric_states:
                    current_rubric_states[rubric_idx]["justification"] = result.get(
                        "justification", "No justification provided"
                    )
                    current_rubric_states[rubric_idx]["meetExpectation"] = result.get(
                        "meetExpectation", current_rubric_states[rubric_idx]["meetExpectation"]
                    )

    final_nl_rubric_checks = evaluator_traj.TrajectoryEvaluator._convert_states_to_checks(current_rubric_states)
    all_expectations_met = all(r.met for r in final_nl_rubric_checks) and len(final_nl_rubric_checks) > 0
    rubric_score = (
        sum(1.0 if r.met else 0.0 for r in final_nl_rubric_checks) / len(final_nl_rubric_checks)
        if final_nl_rubric_checks else 0.0
    )
    reward = 1.0 if all_expectations_met else 0.0
    return RewardInfo(
        reward=reward,
        nl_rubrics=final_nl_rubric_checks,
        reward_breakdown={RewardType.NL_ASSERTION: rubric_score},
        info={"evaluation_method": "sliding_window_batched"},
    )


def create_vitabench_batch_episodes(
    dialogues_batch: List[Dict],
    config,
    debug: bool = False,
    language: str = "english",
) -> List[EpisodeState]:
    """
    Create N VitaBench episodes sharing one batch env (one model load, one BatchVitaBenchEnvironment).
    Use when vitabench_user_is_local(config) so we operate at batch width and avoid N separate env inits.
    """
    if not dialogues_batch:
        return []
    if not vitabench_user_is_local(config):
        raise ValueError("create_vitabench_batch_episodes requires local user model")
    from .vitabench_env import BatchVitaBenchEnvironment, EpisodeVitaBenchEnvWrapper

    # First episode: full init (model load + monkey-patch)
    first = dialogues_batch[0]
    task_id = first.get("id", str(0))
    domain = first.get("domain", "ota")
    ep0 = create_vitabench_episode_state(
        task_id=task_id,
        domain=domain,
        config=config,
        debug=debug,
        language=language,
    )
    user_model = ep0.env.user_model
    user_tokenizer = ep0.env.user_tokenizer
    evaluator_model = getattr(ep0.env, "evaluator_model", None)
    evaluator_tokenizer = getattr(ep0.env, "evaluator_tokenizer", None)
    preloaded = (user_model, user_tokenizer, evaluator_model, evaluator_tokenizer)
    slots = [ep0.env]
    episodes = [ep0]
    for i in range(1, len(dialogues_batch)):
        d = dialogues_batch[i]
        ep = create_vitabench_episode_state(
            task_id=d.get("id", str(i)),
            domain=d.get("domain", "ota"),
            config=config,
            debug=debug,
            language=language,
            _preloaded=preloaded,
        )
        slots.append(ep.env)
        episodes.append(ep)

    batch_env = BatchVitaBenchEnvironment(
        slots, user_model, user_tokenizer,
        evaluator_model=evaluator_model,
        evaluator_tokenizer=evaluator_tokenizer,
    )
    # Replace each episode.env with wrapper so all share batch_env
    for i, ep in enumerate(episodes):
        ep.env = EpisodeVitaBenchEnvWrapper(batch_env, i)

    return episodes


def batched_full_traj_judge(
    tasks,
    trajectories,
    final_states,
    llm_evaluator,
    llm_args_evaluator,
    language,
    model,
    tokenizer,
) -> List:
    """
    Batched full-trajectory judge for VitaBench with a local evaluator model.

    This mirrors VitaBench's `TrajectoryEvaluator.calculate_reward_full_traj_rubric`,
    but evaluates multiple (task, trajectory, final_state) triplets in a single
    batched forward pass through the local evaluator model.
    """
    from vita.evaluator import evaluator_traj
    from vita.utils import evaluator_extracter, get_weekday
    from vita.prompts import get_prompts
    from vita.data_model.message import SystemMessage, UserMessage
    from vita.data_model.simulation import RewardInfo, RewardType

    if model is None or tokenizer is None:
        raise ValueError("batched_full_traj_judge requires a local evaluator model and tokenizer.")

    if llm_evaluator is None:
        raise ValueError("batched_full_traj_judge requires llm_evaluator to be specified.")

    if llm_args_evaluator is None:
        llm_args_evaluator = {}

    if len(tasks) != len(trajectories) or len(tasks) != len(final_states):
        raise ValueError("tasks, trajectories, and final_states must have the same length.")

    rewards: List[RewardInfo] = []
    prompts_obj = get_prompts(language)

    prompts_batch: List[str] = []
    metadata = []

    for idx in range(len(tasks)):
        task = tasks[idx]
        trajectory = trajectories[idx]
        final_state = final_states[idx]

        evaluation_criteria = task.evaluation_criteria
        if evaluation_criteria is None:
            rewards.append(
                RewardInfo(
                    reward=1.0,
                    nl_rubrics=[],
                    reward_breakdown={RewardType.NL_ASSERTION: 1.0},
                    info={"note": "No evaluation criteria"},
                )
            )
            continue

        if not evaluation_criteria.expected_states and not evaluation_criteria.overall_rubrics:
            rewards.append(
                RewardInfo(
                    reward=1.0,
                    nl_rubrics=[],
                    reward_breakdown={RewardType.NL_ASSERTION: 1.0},
                    info={"note": "No rubric to evaluate"},
                )
            )
            continue

        env_info = {
            "system_time": "",
            "database": []
        }

        if task.environment:
            time_str = task.environment.get("time", "")
            if time_str:
                weekday = get_weekday(time_str, language)
                env_info["system_time"] = f"{time_str} {weekday or ''}"

        current_rubric_states = evaluator_traj.TrajectoryEvaluator._initialize_rubric_states(evaluation_criteria)
        trajectory_content = evaluator_traj.TrajectoryEvaluator._format_window_content(trajectory)
        current_rubrics_str = evaluator_traj.TrajectoryEvaluator._format_current_rubrics(current_rubric_states)

        system_prompt = prompts_obj.full_trajectory_eval_template.format(
            env_info=env_info,
            user_instruction=task.instructions
        )

        user_prompt = f"""
    # Input
    <trajectory_content>
    {trajectory_content}
    </trajectory_content>

    <current_rubrics>
    {current_rubrics_str}
    </current_rubrics>
    """

        full_prompt = messages_to_prompt([
            SystemMessage(role="system", content=system_prompt),
            UserMessage(role="user", content=user_prompt),
        ])
        prompts_batch.append(full_prompt)
        metadata.append({
            "current_rubric_states": current_rubric_states,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        })

    if prompts_batch:
        max_new_tokens = llm_args_evaluator.get("max_tokens", 256)
        temperature = llm_args_evaluator.get("temperature", 0.7)
        do_sample = temperature > 0.0

        generated_texts = batch_generate(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts_batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            chunk_size=llm_args_evaluator.get("vitabench_minibatch_size")
        )

        if len(generated_texts) != len(prompts_batch):
            raise RuntimeError(
                f"batched_full_traj_judge: expected {len(prompts_batch)} completions, "
                f"got {len(generated_texts)}."
            )

        for generated_text, meta in zip(generated_texts, metadata):
            current_rubric_states = meta["current_rubric_states"]
            updated_states = copy.deepcopy(current_rubric_states)

            result_data = evaluator_extracter(generated_text)
            # Local models may return a JSON object (dict) instead of a list; iterating yields keys (str).
            if isinstance(result_data, dict):
                result_data = [dict(rubric_idx=k, **v) for k, v in result_data.items()]
            if result_data:
                for result in result_data:
                    rubric_idx = result.get("rubric_idx")
                    if rubric_idx and rubric_idx in updated_states:
                        updated_states[rubric_idx]["justification"] = result.get(
                            "justification",
                            "No justification provided"
                        )
                        updated_states[rubric_idx]["meetExpectation"] = result.get(
                            "meetExpectation",
                            updated_states[rubric_idx]["meetExpectation"]
                        )

            final_nl_rubric_checks = evaluator_traj.TrajectoryEvaluator._convert_states_to_checks(updated_states)
            all_expectations_met = (
                all(result.met for result in final_nl_rubric_checks) and len(final_nl_rubric_checks) > 0
            )
            rubric_score = (
                sum(1.0 if result.met else 0.0 for result in final_nl_rubric_checks) /
                len(final_nl_rubric_checks) if final_nl_rubric_checks else 0.0
            )
            reward = 1.0 if all_expectations_met else 0.0

            rewards.append(
                RewardInfo(
                    reward=reward,
                    nl_rubrics=final_nl_rubric_checks,
                    reward_breakdown={RewardType.NL_ASSERTION: rubric_score},
                    info={"evaluation_method": "full_trajectory_with_rubrics_batched"},
                )
            )

    return rewards

