"""Episode creation registry and context."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..training.episode_state import EpisodeState


@dataclass
class EpisodeCreationContext:
    """Structured context for episode creation."""

    dialogue_idx: int
    dialogue_data: Any
    config: Any
    debug: bool = False
    user_agent: Optional[Any] = None
    persona: Optional[Dict] = None
    available_personas: Optional[List[Dict]] = None
    patient_agent: Optional[Any] = None
    reward_model: Optional[Any] = None
    reward_tokenizer: Optional[Any] = None
    judge_model: Optional[Any] = None
    judge_tokenizer: Optional[Any] = None
    multiwoz_mode: bool = False
    environment_type: str = "multiwoz"


def _create_vitabench_episode(ctx: EpisodeCreationContext) -> EpisodeState:
    from .vitabench_helpers import create_vitabench_episode_state

    if isinstance(ctx.dialogue_data, dict):
        task_id = ctx.dialogue_data.get("id", str(ctx.dialogue_idx))
        domain = ctx.dialogue_data.get("domain", "ota")
    else:
        task_id = ctx.dialogue_data.id
        domain = ctx.dialogue_data.domain

    return create_vitabench_episode_state(
        task_id=task_id,
        domain=domain,
        config=ctx.config,
        debug=ctx.debug,
        language=getattr(ctx.config, "language", "english"),
    )


def _create_salesagent_episode(ctx: EpisodeCreationContext) -> EpisodeState:
    from .salesagent_helpers import create_salesagent_episode_state

    conversation = (
        ctx.dialogue_data
        if isinstance(ctx.dialogue_data, dict)
        else ctx.dialogue_data
    )
    return create_salesagent_episode_state(
        conversation_idx=ctx.dialogue_idx,
        conversation=conversation,
        config=ctx.config,
        debug=ctx.debug,
    )


def _create_userbench_episode(ctx: EpisodeCreationContext) -> EpisodeState:
    from .userbench_helpers import create_userbench_episode_state

    task_data = (
        ctx.dialogue_data
        if isinstance(ctx.dialogue_data, dict)
        else ctx.dialogue_data
    )
    return create_userbench_episode_state(
        task_data=task_data,
        config=ctx.config,
        debug=ctx.debug,
    )


def _create_cares_episode(ctx: EpisodeCreationContext) -> EpisodeState:
    from .cares_helpers import create_cares_episode_state

    return create_cares_episode_state(
        example_idx=ctx.dialogue_idx,
        example=ctx.dialogue_data,
        config=ctx.config,
        debug=ctx.debug,
        patient_agent=ctx.patient_agent,
        reward_model=ctx.reward_model,
        reward_tokenizer=ctx.reward_tokenizer,
        judge_model=ctx.judge_model,
        judge_tokenizer=ctx.judge_tokenizer,
    )


def _create_wildjailbreak_episode(ctx: EpisodeCreationContext) -> EpisodeState:
    from .wildjailbreak_helpers import create_wildjailbreak_episode_state

    return create_wildjailbreak_episode_state(
        example_idx=ctx.dialogue_idx,
        example=ctx.dialogue_data,
        config=ctx.config,
        debug=ctx.debug,
        patient_agent=ctx.patient_agent,
        reward_model=ctx.reward_model,
        reward_tokenizer=ctx.reward_tokenizer,
        judge_model=ctx.judge_model,
        judge_tokenizer=ctx.judge_tokenizer,
    )


def _create_redbench_episode(ctx: EpisodeCreationContext) -> EpisodeState:
    from .redbench_helpers import create_redbench_episode_state

    return create_redbench_episode_state(
        example_idx=ctx.dialogue_idx,
        example=ctx.dialogue_data,
        config=ctx.config,
        debug=ctx.debug,
        patient_agent=ctx.patient_agent,
        reward_model=ctx.reward_model,
        reward_tokenizer=ctx.reward_tokenizer,
        judge_model=ctx.judge_model,
        judge_tokenizer=ctx.judge_tokenizer,
    )


def _create_harmbench_episode(ctx: EpisodeCreationContext) -> EpisodeState:
    from .harmbench_helpers import create_harmbench_episode_state

    return create_harmbench_episode_state(
        example_idx=ctx.dialogue_idx,
        example=ctx.dialogue_data,
        config=ctx.config,
        debug=ctx.debug,
        patient_agent=ctx.patient_agent,
        reward_model=ctx.reward_model,
        reward_tokenizer=ctx.reward_tokenizer,
        judge_model=ctx.judge_model,
        judge_tokenizer=ctx.judge_tokenizer,
    )


EPISODE_CREATORS: Dict[str, Callable[[EpisodeCreationContext], EpisodeState]] = {
    "vitabench": _create_vitabench_episode,
    "salesagent": _create_salesagent_episode,
    "userbench": _create_userbench_episode,
    "cares": _create_cares_episode,
    "wildjailbreak": _create_wildjailbreak_episode,
    "redbench": _create_redbench_episode,
    "harmbench": _create_harmbench_episode,
}


def create_episode_from_registry(ctx: EpisodeCreationContext) -> EpisodeState:
    """
    Create episode by looking up creator in registry.

    MultiWOZ (multiwoz, multiwoz_offline, multiwoz_online) is handled separately
    in main.create_episode_state due to its distinct logic.
    """
    creator = EPISODE_CREATORS.get(ctx.environment_type)
    if creator is None:
        raise NotImplementedError(
            f"Unsupported environment type: {ctx.environment_type}"
        )
    return creator(ctx)
