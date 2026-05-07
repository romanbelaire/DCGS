# Environments module

from .dialogue_env import DialogueEnvironment, StepResult
from .multiwoz_env import MultiWOZEnvironment, EnvironmentState
from .online_env import OnlineEnvironment, OnlineEnvironmentState
from .salesagent_env import SalesAgentEnvironment
from .userbench_env import UserBenchEnvironment
from .cares_env import CARESOfflineEnvironment, CARESOnlineEnvironment

# VitaBench is imported lazily so that runs with environment_type != "vitabench"
# do not require vitabench_env.py (e.g. different checkouts or shared codebase).
# Both ablation 1 and 3 use environment_type "vitabench" and hit the same code path;
# the only way one can fail and the other succeed is different codebase state when
# each job ran (e.g. different node, or file added/removed between runs).


def __getattr__(name: str):
    if name == "VitaBenchEnvironment":
        from .vitabench_env import VitaBenchEnvironment
        return VitaBenchEnvironment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DialogueEnvironment",
    "StepResult",
    "MultiWOZEnvironment",
    "EnvironmentState",
    "OnlineEnvironment",
    "OnlineEnvironmentState",
    "VitaBenchEnvironment",
    "SalesAgentEnvironment",
    "UserBenchEnvironment",
    "CARESOfflineEnvironment",
    "CARESOnlineEnvironment",
]

