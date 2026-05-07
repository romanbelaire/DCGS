"""Base configuration class for the LLM Context Belief Framework."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class BaseConfig:
    """Type-safe configuration with validation."""
    
    # Model settings
    model_name: str = "HuggingFaceH4/zephyr-7b-beta"
    user_model_name: Optional[str] = None  # If None, uses same model as model_name
    device: str = "cuda"
    use_bf16: bool = True
    
    # GPT agent settings (for using OpenAI API for agents instead of local model)
    use_gpt_for_agents: bool = False  # If True, use GPT API for high-level and low-level agents
    use_gpt_for_user: bool = False  # If True, use GPT API for user agent (in addition to agents)
    gpt_agent_model: str = "gpt-4o-mini"  # GPT model to use for agents (e.g., "gpt-4o-mini", "gpt-4o")
    gpt_reasoning_effort: Optional[str] = None  # GPT reasoning effort for gpt-5 family (e.g., "none", "medium")
    gpt_agent_api_key: Optional[str] = None  # OpenAI API key (if None, uses OPENAI_API_KEY env var)
    hl_enable_thinking: Optional[bool] = None  # If None, defaults to True for Qwen3.5 and False otherwise
    ll_enable_thinking: Optional[bool] = None  # If None, defaults to False
    
    # Agent hyperparameters
    n_candidates: int = 5
    n_ll_candidates: int = 5  # K LL candidates per selected belief; >1 enables token critic selection
    n_next_state_candidates: int = 3  # Fewer candidates for next-state Q-value computation (faster)
    belief_gen_temperature: float = 0.7
    max_tokens: int = 256  
    epsilon: float = 0.1  # For epsilon-greedy action selection
    belief_gen_template: str = "belief_generation"  # Template name for belief generation (options: belief_generation, belief_generation_v2, belief_generation_v3, belief_generation_v4, belief_generation_v5)
    
    # Belief update parameters
    dpo_temperature: float = 1.0  # Fixed DPO temperature
    
    # Value function parameters
    discount_factor: float = 0.9
    learning_rate: float = 1e-4
    entropy_coef: float = 0.1
    
    # Marginal token rewards parameters
    use_marginal_token_rewards: bool = False  # Enable marginal token reward computation
    num_masked_marginal_tokens: int = 5  # Number of top-k tokens to process
    pairwise_marginal_masking: bool = True  # Enable pairwise masking
    max_marginal_token_positions: int = 10  # Memory limit for token processing
    use_task_reward: bool = True  # Include instruction-following reward
    reward_model_type: Optional[str] = "llamaguard"  # "llamaguard", "shieldgemma", "skywork", or None (None = use environment reward)
    reward_model_name: Optional[str] = None  # Path to reward model (if None, uses sequence critic)
    marginal_reward_chunk_size: int = 4  # Chunk size for processing masked versions
    token_level_reward_schema: str = "marginal"  # "marginal" | "uniform" | "decayed" (token critic targets)
    token_reward_decay_gamma: Optional[float] = None  # For "decayed"; None = use discount_factor

    # Contrastive learning parameters
    contrastive_coef: float = 0.0  # Coefficient for contrastive loss (0.0 = disabled)
    contrastive_ablation_mode: str = "none"  # One of: "crossed_data", "random_noise", "noise_in_candidates", "crossed_data_in_candidates", "none"

    # Regret critic (adversarial robustness) - for cares/wildjailbreak
    use_regret_critic: bool = False  # Enable three-critic: value, min-value, regret
    regret_critic_beta: float = 0.2  # High-level policy: softmax over (1-beta)*Q - beta*Q^regret (default 0.8Q - 0.2 Q^regret)
    regret_zero_sum_targets: bool = True  # Enforce zero-sum regret targets
    regret_min_target_mode: str = "sampled_q_min"  # sampled_q_min: min over sampled high-level candidates
    
    # Low-level action generation mode
    ll_action_belief_only: bool = True  # If True, LL actions depend only on belief P(a|b). If False, use P(a|o,b) (old mode for ablations)
    defender_backend: str = "standard"  # "standard" | "smoothllm" | "tpo"

    # SmoothLLM defender hyperparameters
    pert_type: str = "RandomSwapPerturbation"
    pert_pct: float = 10.0
    num_copies: int = 8
    smooth_batch_size: int = 8

    # TPO defender hyperparameters
    tpo_sample_size: int = 8
    tpo_max_iters: int = 3
    tpo_temperature: float = 0.7
    tpo_reward_model: str = "heuristic"
    tpo_mode: str = "tpo"  # "tpo" | "revision" | "bon"
    
    # Baseline mode (skip high-level belief generation, use only conversation history)
    baseline_mode: bool = False  # If True, skip high-level agent and DST, use only conversation history for low-level actions
    
    # Random belief selection (skip Q-value computation, always randomly select from candidates)
    random_belief_selection: bool = False  # If True, always randomly selects from candidates (no Q-value computation, no value function)

    # Hierarchical setup
    use_hierarchical_agent: bool = False
    high_level_policy_type: str = "belief_candidates"  # "belief_candidates" | "freeform"
    critic_only_training: bool = False
    freeform_n_instructions: int = 5
    freeform_temperature: float = 0.8
    freeform_top_p: float = 0.9
    freeform_max_new_tokens: int = 96
    freeform_iterative_candidate_generation: bool = False
    freeform_per_instruction_max_new_tokens: int = 96
    hierarchical_rejection_sampling: bool = True
    hierarchical_rejection_candidates: int = 3
    
    # Training settings
    max_turns: int = 20
    batch_size: int = 16
    episode_batch_size: int = 1  # Number of episodes to process in parallel (all steps batched)
    vitabench_judge_batch_size: int = 16  # Number of VitaBench episodes to judge together with local evaluator
    evaluation_interval: int = 100  # Episodes between evaluation runs
    transition_prob_chunk_size: int = 8  # Chunk size for transition probability computation to prevent OOM
    q_value_chunk_size: int = 16  # Chunk size for Q-value computation to prevent OOM
    batch_generation_chunk_size: int = 16  # Chunk size for batch text generation to prevent OOM
    n_action_candidates: int = 5  # Number of candidate low-level actions to generate per belief for P(a|b) normalization
    output_dir: str = "outputs"
    
    # Online Q-learning settings
    online_batch_size: int = 8  # Batch size for online updates (from replay buffer)
    replay_buffer_size: int = 1024  # Size of replay buffer for online mode
    epsilon_decay_rate: float = 0.995  # Epsilon decay per episode (exponential)
    epsilon_min: float = 0.05  # Minimum epsilon value
    online_update_frequency: int = 1  # Update Q-function every N turns (1 = every turn, 2 = every other turn, etc.)
    
    # Data settings
    data_path: Optional[str] = None
    persona_dir: str = "src/prompts/personas"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    evaluation_sample_size: Optional[int] = None  # Subsample size for evaluation (None = use all)
    max_dialogues: Optional[int] = None  # Maximum number of dialogues to use (None = use all, for faster training set to e.g. 2000)
    
    # Environment settings
    environment_type: str = "multiwoz_offline"  # "multiwoz_offline", "multiwoz_online", "vitabench", "salesagent", "userbench", "cares", "wildjailbreak", "redbench", "harmbench"
    language: Optional[str] = None  # Language for VitaBench (e.g., "english")
    task_config: Dict = field(default_factory=dict)

    # CARES-18K settings
    cares_split: str = "train"
    cares_use_lead_in: bool = True
    cares_online: bool = True  # Use PatientAgent (online) vs synthetic lead-in (offline)

    # WildJailbreak settings
    wildjailbreak_split: str = "train"
    wildjailbreak_data_types: Optional[List[str]] = None  # None = all 4; else e.g. ["vanilla_harmful", "adversarial_benign"]
    wildjailbreak_use_lead_in: bool = True
    wildjailbreak_online: bool = True  # Use PatientAgent (online) vs synthetic lead-in (offline)

    # RedBench settings
    redbench_split: str = "train"
    redbench_subsets: Optional[List[str]] = None  # None = all subsets in knoveleng/redbench
    redbench_max_examples: Optional[int] = None  # None = all rows across selected subsets
    redbench_mapping_mode: str = "category"  # "category" | "source" | "category_with_source_override"
    redbench_refusal_sources: Optional[List[str]] = None  # None = default refusal sources (CoCoNot, ORBench, SGXSTest, XSTest)
    redbench_use_lead_in: bool = True
    redbench_online: bool = True  # Use PatientAgent (online) vs synthetic lead-in (offline)

    # HarmBench settings
    harmbench_split: str = "train"
    harmbench_max_examples: Optional[int] = None  # None = all rows in RedBench HarmBench subset
    harmbench_use_lead_in: bool = True
    harmbench_online: bool = True  # Use PatientAgent (online) vs synthetic lead-in (offline)
    
    # VitaBench LLM settings
    vitabench_llm_user: Optional[str] = None  # LLM model for VitaBench user simulator (None = use default)
    vitabench_llm_evaluator: Optional[str] = None  # LLM model for VitaBench evaluator/judge (None = use default)
    vitabench_llm_args_user: Optional[Dict] = None  # Additional args for user LLM
    vitabench_llm_args_evaluator: Optional[Dict] = None  # Additional args for evaluator LLM
    vitabench_minibatch_size: int = 8  # Minibatch size for local VitaBench batch_generate
    openai_api_key: Optional[str] = None  # OpenAI API key (if None, uses OPENAI_API_KEY env var)
    
    # Checkpoint settings
    checkpoint_path: Optional[str] = None  # Path to pretrained Q-network checkpoint
    
    # Debug settings
    debug: bool = False
    
    def __post_init__(self) -> None:
        if self.hl_enable_thinking not in (True, False):
            self.hl_enable_thinking = self.model_name.startswith("Qwen/Qwen3.5")
        if self.ll_enable_thinking not in (True, False):
            self.ll_enable_thinking = False
        if self.evaluation_interval <= 0:
            raise ValueError("evaluation_interval must be a positive integer")
        valid_ablation_modes = ["crossed_data", "random_noise", "noise_in_candidates", "crossed_data_in_candidates", "none"]
        if self.contrastive_ablation_mode not in valid_ablation_modes:
            raise ValueError(f"contrastive_ablation_mode must be one of {valid_ablation_modes}, got {self.contrastive_ablation_mode}")
        valid_token_reward_schemas = ["marginal", "uniform", "decayed"]
        if self.token_level_reward_schema not in valid_token_reward_schemas:
            raise ValueError(
                f"token_level_reward_schema must be one of {valid_token_reward_schemas}, got {self.token_level_reward_schema!r}"
            )
        valid_backends = ["standard", "smoothllm", "tpo"]
        if self.defender_backend not in valid_backends:
            raise ValueError(f"defender_backend must be one of {valid_backends}, got {self.defender_backend}")
        valid_high_level_policy_types = ["belief_candidates", "freeform"]
        if self.high_level_policy_type not in valid_high_level_policy_types:
            raise ValueError(
                f"high_level_policy_type must be one of {valid_high_level_policy_types}, got {self.high_level_policy_type}"
            )
        if self.use_hierarchical_agent and self.high_level_policy_type != "freeform":
            raise ValueError(
                "use_hierarchical_agent requires high_level_policy_type='freeform' for the hierarchical setup."
            )
        if self.use_hierarchical_agent and not self.critic_only_training:
            raise ValueError(
                "use_hierarchical_agent requires critic_only_training=True in this codebase."
            )
        if self.use_hierarchical_agent and self.use_gpt_for_agents:
            raise ValueError(
                "use_hierarchical_agent requires local high-level and low-level agents. Set use_gpt_for_agents=False."
            )
        if self.freeform_n_instructions <= 0:
            raise ValueError("freeform_n_instructions must be positive.")
        if self.freeform_per_instruction_max_new_tokens <= 0:
            raise ValueError("freeform_per_instruction_max_new_tokens must be positive.")
        if self.hierarchical_rejection_candidates <= 0:
            raise ValueError("hierarchical_rejection_candidates must be positive.")
        if self.n_ll_candidates < 1:
            raise ValueError("n_ll_candidates must be >= 1.")
        valid_regret_min_target_modes = ["sampled_q_min"]
        if self.regret_min_target_mode not in valid_regret_min_target_modes:
            raise ValueError(
                f"regret_min_target_mode must be one of {valid_regret_min_target_modes}, got {self.regret_min_target_mode}"
            )
        if self.use_regret_critic and not self.regret_zero_sum_targets:
            raise ValueError(
                "use_regret_critic requires regret_zero_sum_targets=True for zero-sum alignment."
            )
        if self.use_regret_critic and self.regret_min_target_mode != "sampled_q_min":
            raise ValueError(
                "use_regret_critic requires regret_min_target_mode='sampled_q_min'."
            )
        if self.use_regret_critic and not math.isclose(self.regret_critic_beta, 0.2, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "use_regret_critic requires regret_critic_beta=0.2 (softmax policy scores 0.8*Q - 0.2*Q^regret)."
            )
        valid_redbench_mapping_modes = [
            "category",
            "source",
            "category_with_source_override",
        ]
        if self.redbench_mapping_mode not in valid_redbench_mapping_modes:
            raise ValueError(
                f"redbench_mapping_mode must be one of {valid_redbench_mapping_modes}, got {self.redbench_mapping_mode}"
            )
    
    @classmethod
    def from_json(cls, config_path: str) -> "BaseConfig":
        """Load configuration from JSON file."""
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = {
            "model_name": self.model_name,
            "user_model_name": self.user_model_name,
            "device": self.device,
            "use_bf16": self.use_bf16,
            "use_gpt_for_agents": getattr(self, 'use_gpt_for_agents', False),
            "use_gpt_for_user": getattr(self, 'use_gpt_for_user', False),
            "gpt_agent_model": getattr(self, 'gpt_agent_model', 'gpt-4o-mini'),
            "gpt_reasoning_effort": getattr(self, 'gpt_reasoning_effort', None),
            "gpt_agent_api_key": getattr(self, 'gpt_agent_api_key', None),
            "hl_enable_thinking": self.hl_enable_thinking,
            "ll_enable_thinking": self.ll_enable_thinking,
            "n_candidates": self.n_candidates,
            "n_ll_candidates": self.n_ll_candidates,
            "belief_gen_temperature": self.belief_gen_temperature,
            "max_tokens": self.max_tokens,
            "epsilon": self.epsilon,
            "belief_gen_template": self.belief_gen_template,
            "dpo_temperature": self.dpo_temperature,
            "discount_factor": self.discount_factor,
            "learning_rate": self.learning_rate,
            "entropy_coef": self.entropy_coef,
            "contrastive_coef": self.contrastive_coef,
            "contrastive_ablation_mode": self.contrastive_ablation_mode,
            "use_regret_critic": getattr(self, 'use_regret_critic', False),
            "regret_critic_beta": getattr(self, 'regret_critic_beta', 0.2),
            "regret_zero_sum_targets": getattr(self, 'regret_zero_sum_targets', True),
            "regret_min_target_mode": getattr(self, 'regret_min_target_mode', 'sampled_q_min'),
            "ll_action_belief_only": self.ll_action_belief_only,
            "defender_backend": self.defender_backend,
            "pert_type": self.pert_type,
            "pert_pct": self.pert_pct,
            "num_copies": self.num_copies,
            "smooth_batch_size": self.smooth_batch_size,
            "tpo_sample_size": self.tpo_sample_size,
            "tpo_max_iters": self.tpo_max_iters,
            "tpo_temperature": self.tpo_temperature,
            "tpo_reward_model": self.tpo_reward_model,
            "tpo_mode": self.tpo_mode,
            "baseline_mode": getattr(self, 'baseline_mode', False),
            "random_belief_selection": getattr(self, 'random_belief_selection', False),
            "use_hierarchical_agent": self.use_hierarchical_agent,
            "high_level_policy_type": self.high_level_policy_type,
            "critic_only_training": self.critic_only_training,
            "freeform_n_instructions": self.freeform_n_instructions,
            "freeform_temperature": self.freeform_temperature,
            "freeform_top_p": self.freeform_top_p,
            "freeform_max_new_tokens": self.freeform_max_new_tokens,
            "freeform_iterative_candidate_generation": self.freeform_iterative_candidate_generation,
            "freeform_per_instruction_max_new_tokens": self.freeform_per_instruction_max_new_tokens,
            "hierarchical_rejection_sampling": self.hierarchical_rejection_sampling,
            "hierarchical_rejection_candidates": self.hierarchical_rejection_candidates,
            "max_turns": self.max_turns,
            "batch_size": self.batch_size,
            "episode_batch_size": self.episode_batch_size,
            "vitabench_judge_batch_size": self.vitabench_judge_batch_size,
            "vitabench_minibatch_size": self.vitabench_minibatch_size,
            "online_batch_size": self.online_batch_size,
            "replay_buffer_size": self.replay_buffer_size,
            "epsilon_decay_rate": self.epsilon_decay_rate,
            "epsilon_min": self.epsilon_min,
            "output_dir": self.output_dir,
            "evaluation_interval": self.evaluation_interval,
            "transition_prob_chunk_size": self.transition_prob_chunk_size,
            "q_value_chunk_size": self.q_value_chunk_size,
            "batch_generation_chunk_size": self.batch_generation_chunk_size,
            "data_path": self.data_path,
            "persona_dir": self.persona_dir,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "evaluation_sample_size": self.evaluation_sample_size,
            "max_dialogues": self.max_dialogues,
            "environment_type": self.environment_type,
            "language": self.language,
            "task_config": self.task_config,
            "cares_split": getattr(self, 'cares_split', 'train'),
            "cares_use_lead_in": getattr(self, 'cares_use_lead_in', True),
            "cares_online": getattr(self, 'cares_online', True),
            "wildjailbreak_split": getattr(self, 'wildjailbreak_split', 'train'),
            "wildjailbreak_data_types": getattr(self, 'wildjailbreak_data_types', None),
            "wildjailbreak_use_lead_in": getattr(self, 'wildjailbreak_use_lead_in', True),
            "wildjailbreak_online": getattr(self, 'wildjailbreak_online', True),
            "redbench_split": getattr(self, 'redbench_split', 'train'),
            "redbench_subsets": getattr(self, 'redbench_subsets', None),
            "redbench_max_examples": getattr(self, 'redbench_max_examples', None),
            "redbench_mapping_mode": getattr(self, 'redbench_mapping_mode', 'category'),
            "redbench_refusal_sources": getattr(self, 'redbench_refusal_sources', None),
            "redbench_use_lead_in": getattr(self, 'redbench_use_lead_in', True),
            "redbench_online": getattr(self, 'redbench_online', True),
            "harmbench_split": getattr(self, 'harmbench_split', 'train'),
            "harmbench_max_examples": getattr(self, 'harmbench_max_examples', None),
            "harmbench_use_lead_in": getattr(self, 'harmbench_use_lead_in', True),
            "harmbench_online": getattr(self, 'harmbench_online', True),
            "checkpoint_path": self.checkpoint_path,
            "use_marginal_token_rewards": self.use_marginal_token_rewards,
            "num_masked_marginal_tokens": self.num_masked_marginal_tokens,
            "pairwise_marginal_masking": self.pairwise_marginal_masking,
            "max_marginal_token_positions": self.max_marginal_token_positions,
            "use_task_reward": self.use_task_reward,
            "reward_model_type": self.reward_model_type,
            "reward_model_name": self.reward_model_name,
            "marginal_reward_chunk_size": self.marginal_reward_chunk_size,
            "token_level_reward_schema": self.token_level_reward_schema,
            "token_reward_decay_gamma": self.token_reward_decay_gamma,
            "n_next_state_candidates": self.n_next_state_candidates,
            "n_action_candidates": self.n_action_candidates,
            "online_batch_size": self.online_batch_size,
            "replay_buffer_size": self.replay_buffer_size,
            "epsilon_decay_rate": self.epsilon_decay_rate,
            "epsilon_min": self.epsilon_min,
            "online_update_frequency": self.online_update_frequency,
            "vitabench_llm_user": getattr(self, 'vitabench_llm_user', None),
            "vitabench_llm_evaluator": getattr(self, 'vitabench_llm_evaluator', None),
            "vitabench_llm_args_user": getattr(self, 'vitabench_llm_args_user', None),
            "vitabench_llm_args_evaluator": getattr(self, 'vitabench_llm_args_evaluator', None),
            "openai_api_key": getattr(self, 'openai_api_key', None),
            "debug": self.debug,
        }
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

