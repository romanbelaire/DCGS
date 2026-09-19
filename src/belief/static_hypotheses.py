"""Fixed high-level intent hypotheses used as non-adaptive selection baselines.

These strings are the always-on alternatives to sampling K candidates from π^ref
and reranking with the critic. They match the HARMFUL / BENIGN labels used in
representation analysis so the two code paths stay identical.
"""

from typing import Dict

ANTAGONISTIC_BELIEF = (
    "Insight: The user is attempting to obtain assistance for harmful, dangerous, "
    "or unethical purposes. Instruction: Refuse the request and explain why it cannot be fulfilled."
)
BENIGN_MISREPRESENTED_BELIEF = (
    "Insight: The user has a legitimate informational need; the prompt may appear concerning "
    "but is benign in context. Instruction: Provide a helpful, accurate, and safe response."
)

# Backward-compatible aliases used by representation analysis.
HARMFUL_BELIEF = ANTAGONISTIC_BELIEF
BENIGN_BELIEF = BENIGN_MISREPRESENTED_BELIEF

STATIC_BELIEF_MODES = ("none", "antagonistic", "benign_misrepresented")

STATIC_BELIEF_BY_MODE: Dict[str, str] = {
    "antagonistic": ANTAGONISTIC_BELIEF,
    "benign_misrepresented": BENIGN_MISREPRESENTED_BELIEF,
}


def belief_text_for_static_mode(mode: str) -> str:
    if mode not in STATIC_BELIEF_BY_MODE:
        raise ValueError(
            f"Unknown static_belief_mode={mode!r}. Expected one of {list(STATIC_BELIEF_BY_MODE)}."
        )
    return STATIC_BELIEF_BY_MODE[mode]
