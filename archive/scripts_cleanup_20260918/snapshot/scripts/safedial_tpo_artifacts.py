"""Pinned local artifacts for the native TPO arm; no model execution."""

import json
from pathlib import Path

from run_safedial_baseline import file_sha256

ACTOR = "HuggingFaceH4/zephyr-7b-beta"
ACTOR_REVISION = "892b3d7a7b1cf10c7a701c60881cd93df615734c"
REWARD = "sfairXC/FsfairX-LLaMA3-RM-v0.1"
REWARD_REVISION = "94fad49f1b3227aa8b566f415a335adb68ec544c"
DEFAULT_LOCK = Path("configs/safedial/tpo.lock.json")


def validate_lock(path, check_hashes=True):
    lock = json.loads(Path(path).read_text())
    if lock.get("schema_version") != 1 or lock.get("method") != "TPO":
        raise ValueError("Unknown TPO artifact lock")
    for kind, repo, revision, architecture in (
        ("actor", ACTOR, ACTOR_REVISION, "MistralForCausalLM"),
        ("reward", REWARD, REWARD_REVISION, "LlamaForSequenceClassification"),
    ):
        item = lock[kind]
        if (item["repo"], item["revision"]) != (repo, revision):
            raise ValueError(f"Unexpected {kind} identity")
        folder = Path(item["path"])
        index = json.loads((folder / "model.safetensors.index.json").read_text())
        required = set(index["weight_map"].values()) | {
            "model.safetensors.index.json", "config.json", "tokenizer_config.json", "tokenizer.json",
        }
        if not required.issubset(item["sha256"]):
            raise ValueError(f"Incomplete {kind} artifact hashes")
        for name, expected in item["sha256"].items():
            if Path(name).name != name or not (folder / name).is_file():
                raise ValueError(f"Missing/invalid {kind} artifact: {name}")
            if check_hashes and file_sha256(folder / name) != expected:
                raise ValueError(f"{kind} hash mismatch: {name}")
        config = json.loads((folder / "config.json").read_text())
        if architecture not in config.get("architectures", []):
            raise ValueError(f"Unexpected {kind} architecture")
    return lock
