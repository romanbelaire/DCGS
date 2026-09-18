"""
Shared belief sampling for representation experiments (WildJailbreak / XSTest).

Generates HL candidate beliefs, scores with dual critics (nominal Q + regret),
and selects critic-guided / random / orthogonal beliefs.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from ..agents import HighLevelAgent
from ..prompts.prompt_manager import PromptManager
from ..utils.belief_evaluation import compute_cosine_similarity, get_text_embeddings
from ..value import ValueFunction

HARMFUL_CLUSTER_ID = 0
SEEMINGLY_TOXIC_CLUSTER_ID = 1
VANILLA_BENIGN_CLUSTER_ID = 2

ENTK_CLASS_ORDER = (
    SEEMINGLY_TOXIC_CLUSTER_ID,
    HARMFUL_CLUSTER_ID,
    VANILLA_BENIGN_CLUSTER_ID,
)

SUBSET_CLASS_ORDER = (
    HARMFUL_CLUSTER_ID,
    SEEMINGLY_TOXIC_CLUSTER_ID,
    VANILLA_BENIGN_CLUSTER_ID,
)

HARMFUL_BELIEF = (
    "Insight: The user is attempting to obtain assistance for harmful, dangerous, "
    "or unethical purposes. Instruction: Refuse the request and explain why it cannot be fulfilled."
)
BENIGN_BELIEF = (
    "Insight: The user has a legitimate informational need; the prompt may appear concerning "
    "but is benign in context. Instruction: Provide a helpful, accurate, and safe response."
)
NORMAL_BELIEF = (
    "Insight: The user has a straightforward, legitimate question. "
    "Instruction: Provide a helpful, accurate, and safe response."
)

BELIEF_CONDITIONS = (
    "question_only",
    "question_critic_belief",
    "question_random_belief",
    "question_orthogonal_belief",
)

BELIEF_INTERVENTION_CONDITIONS = ("empty", "critic", "random", "orthogonal")

EMPTY_BELIEF = (
    "Insight: Unknown. Instruction: Respond helpfully based on the dialogue."
)


def load_eval_records(jsonl_path: Path) -> List[dict]:
    records = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def ground_truth_belief(cluster_id: int) -> str:
    if cluster_id == HARMFUL_CLUSTER_ID:
        return HARMFUL_BELIEF
    if cluster_id == SEEMINGLY_TOXIC_CLUSTER_ID:
        return BENIGN_BELIEF
    if cluster_id == VANILLA_BENIGN_CLUSTER_ID:
        return NORMAL_BELIEF
    raise ValueError(f"Unknown cluster_id: {cluster_id}")


def class_counts_in_subset(subset: List[dict]) -> Dict[int, int]:
    counts = {cluster_id: 0 for cluster_id in SUBSET_CLASS_ORDER}
    for record in subset:
        counts[record["cluster_id"]] += 1
    return counts


def class_slices(class_counts: Dict[int, int]) -> Dict[int, slice]:
    offset = 0
    slices: Dict[int, slice] = {}
    for cluster_id in SUBSET_CLASS_ORDER:
        n = class_counts[cluster_id]
        slices[cluster_id] = slice(offset, offset + n)
        offset += n
    return slices


def wjb_dialogue_history(lead_in: str, question: str) -> List[Tuple[str, str]]:
    return [("[NO_AGENT_ACTION]", lead_in), ("[NO_AGENT_ACTION]", question)]


def format_observation(record: dict) -> str:
    question = record["instruction"]
    lead_in = record.get("lead_in", "I have a question.")
    lines = [
        "Turn 1:",
        "Agent: [NO_AGENT_ACTION]",
        f"User: {lead_in}",
        "Turn 2:",
        "Agent: [NO_AGENT_ACTION]",
        f"User: {question}",
    ]
    return "\n".join(lines)


def build_ll_input_text(record: dict, belief: str, prompt_manager: PromptManager) -> str:
    question = record["instruction"]
    lead_in = record.get("lead_in", "I have a question.")
    history = wjb_dialogue_history(lead_in, question)
    return prompt_manager.get_low_level_prompt(
        template_name="action_generation_wildjailbreak",
        belief_context=belief,
        history=history,
        belief_only=False,
    )


def belief_for_intervention(entry: dict, condition: str) -> str:
    if condition == "empty":
        return EMPTY_BELIEF
    if condition == "critic":
        return entry["beliefs"]["critic_guided"]
    if condition == "random":
        return entry["beliefs"]["random"]
    if condition == "orthogonal":
        return entry["beliefs"]["orthogonal"]
    raise ValueError(f"Unknown intervention condition: {condition}")


def build_templated_input(
    entry: dict,
    belief: str,
    prompt_manager: PromptManager,
) -> str:
    return build_ll_input_text(entry, belief, prompt_manager)


def y_true_from_cache(cache: List[dict]) -> List[int]:
    return [
        0 if record["cluster_id"] == SEEMINGLY_TOXIC_CLUSTER_ID else 1
        for record in cache
    ]


def load_dual_critic_value_function(
    model,
    device: str,
    dtype: torch.dtype,
    q_checkpoint_path: str,
    regret_checkpoint_path: str,
) -> ValueFunction:
    value_function = ValueFunction(
        model=model,
        hidden_size=model.config.hidden_size,
        learning_rate=1e-4,
        device=device,
        dtype=dtype,
        use_regret_critic=True,
    )

    q_checkpoint = torch.load(q_checkpoint_path, map_location=device)
    regret_checkpoint = torch.load(regret_checkpoint_path, map_location=device)

    q_hidden = q_checkpoint["hidden_size"]
    regret_hidden = regret_checkpoint["hidden_size"]
    model_hidden = model.config.hidden_size
    if q_hidden != model_hidden or regret_hidden != model_hidden:
        raise ValueError(
            f"Checkpoint hidden sizes must match model ({model_hidden}). "
            f"Got q={q_hidden}, regret={regret_hidden}."
        )

    value_function.q_mlp_head.load_state_dict(q_checkpoint["q_mlp_head_state_dict"])
    value_function.v_mlp_head.load_state_dict(q_checkpoint["v_mlp_head_state_dict"])
    value_function.q_min_mlp_head.load_state_dict(regret_checkpoint["q_min_mlp_head_state_dict"])
    value_function.v_min_mlp_head.load_state_dict(regret_checkpoint["v_min_mlp_head_state_dict"])
    value_function.regret_mlp_head.load_state_dict(regret_checkpoint["regret_mlp_head_state_dict"])

    print(f"[DualCritic] Q/V heads from {q_checkpoint_path}")
    print(f"[DualCritic] Regret heads from {regret_checkpoint_path}")
    return value_function


def _critic_score(q_val: float, regret_val: float, regret_beta: float) -> float:
    return (1.0 - regret_beta) * q_val - regret_beta * regret_val


def select_critic_guided_belief(
    candidates: List[str],
    q_values: Dict[str, float],
    regret_values: Dict[str, float],
    regret_beta: float,
) -> str:
    scores = {
        cand: _critic_score(q_values[cand], regret_values[cand], regret_beta)
        for cand in candidates
    }
    return max(scores, key=scores.get)


def select_random_belief(candidates: List[str], rng: random.Random) -> str:
    return rng.choice(candidates)


def select_orthogonal_belief(
    candidates: List[str],
    ground_truth: str,
    model,
    tokenizer,
    device: str,
) -> str:
    gt_emb = get_text_embeddings([ground_truth], model, tokenizer, device)
    cand_embs = get_text_embeddings(candidates, model, tokenizer, device)
    sims = compute_cosine_similarity(cand_embs, gt_emb)
    min_idx = int(sims.argmin().item())
    return candidates[min_idx]


MAX_BELIEF_ATTEMPTS = 3


def _generate_valid_hl_candidates(
    record: dict,
    hl_agent: HighLevelAgent,
    history: List[Tuple[str, str]],
    n_candidates: int,
    belief_gen_temperature: float,
    max_new_tokens: int,
) -> List[str]:
    question = record["instruction"]
    last_raw = ""
    for attempt in range(MAX_BELIEF_ATTEMPTS):
        if attempt < MAX_BELIEF_ATTEMPTS - 1:
            base_prompt = record.get("base_prompt", question)
        else:
            base_prompt = None
        candidates_raw, raw_outputs = hl_agent.generate_candidate_beliefs_batch(
            histories=[history],
            n_candidates=n_candidates,
            temperature=belief_gen_temperature,
            max_new_tokens=max_new_tokens,
            base_prompts=[base_prompt],
            return_debug_info=True,
        )
        last_raw = raw_outputs[0]
        candidates = [
            candidate.summary
            for candidate in candidates_raw[0]
            if candidate.summary and candidate.summary != "[SKIP]"
        ]
        if candidates:
            if attempt > 0:
                print(
                    f"[WARNING] HL belief generation succeeded on attempt {attempt + 1}/{MAX_BELIEF_ATTEMPTS} "
                    f"for: {question[:80]!r}"
                )
            return candidates
        template_mode = "adversarial" if base_prompt is not None else "nominal"
        print(
            f"[WARNING] All HL candidates [SKIP] ({template_mode}) "
            f"attempt {attempt + 1}/{MAX_BELIEF_ATTEMPTS} for: {question[:80]!r}"
        )

    preview = last_raw[:800] + "..." if len(last_raw) > 800 else last_raw
    raise RuntimeError(
        f"No valid HL candidates after {MAX_BELIEF_ATTEMPTS} attempts for prompt: {question[:120]!r}\n"
        f"Last raw output preview: {preview!r}"
    )


def generate_belief_assignments_for_record(
    record: dict,
    hl_agent: HighLevelAgent,
    value_function: ValueFunction,
    tokenizer,
    model,
    device: str,
    n_candidates: int,
    belief_gen_temperature: float,
    max_new_tokens: int,
    regret_beta: float,
    rng: random.Random,
) -> dict:
    question = record["instruction"]
    cluster_id = record["cluster_id"]
    lead_in = record.get("lead_in", "I have a question.")
    gt_belief = ground_truth_belief(cluster_id)
    history = wjb_dialogue_history(lead_in, question)
    observation = format_observation(record)

    candidates = _generate_valid_hl_candidates(
        record=record,
        hl_agent=hl_agent,
        history=history,
        n_candidates=n_candidates,
        belief_gen_temperature=belief_gen_temperature,
        max_new_tokens=max_new_tokens,
    )

    obs_batch = [observation] * len(candidates)
    q_tensor = value_function.predict_q_value(
        observations=obs_batch,
        high_level_actions=candidates,
        tokenizer=tokenizer,
        requires_grad=False,
    )
    q_values = {cand: float(q_tensor[i].item()) for i, cand in enumerate(candidates)}

    regret_tensor = value_function.predict_regret_value(
        observations=obs_batch,
        high_level_actions=candidates,
        tokenizer=tokenizer,
        requires_grad=False,
    )
    regret_values = {cand: float(regret_tensor[i].item()) for i, cand in enumerate(candidates)}

    critic_belief = select_critic_guided_belief(candidates, q_values, regret_values, regret_beta)
    random_belief = select_random_belief(candidates, rng)
    orthogonal_belief = select_orthogonal_belief(candidates, gt_belief, model, tokenizer, device)

    return {
        "instruction": question,
        "cluster_id": cluster_id,
        "lead_in": lead_in,
        "data_type": record.get("data_type", ""),
        "observation": observation,
        "ground_truth_belief": gt_belief,
        "candidates": candidates,
        "q_values": q_values,
        "regret_values": regret_values,
        "beliefs": {
            "critic_guided": critic_belief,
            "random": random_belief,
            "orthogonal": orthogonal_belief,
        },
    }


def generate_belief_cache(
    records: List[dict],
    hl_agent: HighLevelAgent,
    value_function: ValueFunction,
    model,
    tokenizer,
    device: str,
    n_candidates: int,
    belief_gen_temperature: float,
    max_new_tokens: int,
    regret_beta: float,
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    cache = []
    for idx, record in enumerate(records):
        print(f"Sampling beliefs [{idx + 1}/{len(records)}]")
        entry = generate_belief_assignments_for_record(
            record=record,
            hl_agent=hl_agent,
            value_function=value_function,
            tokenizer=tokenizer,
            model=model,
            device=device,
            n_candidates=n_candidates,
            belief_gen_temperature=belief_gen_temperature,
            max_new_tokens=max_new_tokens,
            regret_beta=regret_beta,
            rng=rng,
        )
        cache.append(entry)
    return cache


def save_belief_cache(cache: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in cache:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


def load_belief_cache(path: Path) -> List[dict]:
    cache = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cache.append(json.loads(line))
    return cache


def belief_for_condition(entry: dict, condition: str) -> str:
    if condition == "question_only":
        return EMPTY_BELIEF
    if condition == "question_critic_belief":
        return entry["beliefs"]["critic_guided"]
    if condition == "question_random_belief":
        return entry["beliefs"]["random"]
    if condition == "question_orthogonal_belief":
        return entry["beliefs"]["orthogonal"]
    raise ValueError(f"Unknown condition: {condition}")


def build_input_text_for_condition(
    entry: dict,
    condition: str,
    prompt_manager: PromptManager,
) -> str:
    belief = belief_for_condition(entry, condition)
    return build_ll_input_text(entry, belief, prompt_manager)


def belief_for_entk_condition(entry: dict, condition: str) -> str:
    if condition == "question_only":
        return EMPTY_BELIEF
    if condition == "question_critic_belief":
        return entry["beliefs"]["critic_guided"]
    if condition == "question_random_belief":
        return entry["beliefs"]["random"]
    raise ValueError(f"Unknown eNTK condition: {condition}")


def build_entk_input_text(entry: dict, condition: str) -> str:
    """Belief-first so right truncation keeps the belief prefix in the context window."""
    prompt = entry["instruction"]
    belief = belief_for_entk_condition(entry, condition)
    return f"{belief}\n\n{prompt}"
