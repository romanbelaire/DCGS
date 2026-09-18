#!/usr/bin/env python3
"""Stage the official pretrained TPO reward model; reuse the cached Zephyr actor."""

import argparse
import json
from pathlib import Path

from run_safedial_baseline import file_sha256
from safedial_tpo_artifacts import ACTOR, ACTOR_REVISION, REWARD, REWARD_REVISION, DEFAULT_LOCK, validate_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface/hub"))
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    if args.lock.exists():
        validate_lock(args.lock)
        print(f"Existing TPO artifact lock and all hashes PASS: {args.lock}")
        return
    from huggingface_hub import HfApi, snapshot_download
    from safetensors import safe_open
    import torch

    lock = {"schema_version": 1, "method": "TPO"}
    for kind, repo, revision in (("actor", ACTOR, ACTOR_REVISION), ("reward", REWARD, REWARD_REVISION)):
        print(f"Staging {kind}: {repo}@{revision}", flush=True)
        folder = Path(snapshot_download(
            repo, revision=revision, cache_dir=str(args.cache_dir.resolve()),
            allow_patterns=["*.safetensors", "*.json", "tokenizer.model", "README.md"],
            local_files_only=args.offline or kind == "actor", token=False, max_workers=2,
        ))
        index = json.loads((folder / "model.safetensors.index.json").read_text())
        names = set(index["weight_map"].values()) | {
            "model.safetensors.index.json", "config.json", "tokenizer_config.json", "tokenizer.json",
        }
        names.update(p.name for p in folder.iterdir()
                     if p.name in {"tokenizer.model", "special_tokens_map.json", "added_tokens.json", "generation_config.json", "README.md"})
        hashes = {name: file_sha256(folder / name) for name in sorted(names)}
        if kind == "reward" and not args.offline:
            info = HfApi(token=False).model_info(repo, revision=revision, files_metadata=True)
            if info.sha != revision:
                raise ValueError("Reward revision mismatch")
            for item in info.siblings:
                if item.rfilename in hashes and item.lfs:
                    if hashes[item.rfilename] != item.lfs.sha256:
                        raise ValueError(f"Official LFS checksum mismatch: {item.rfilename}")
        lock[kind] = {"repo": repo, "revision": revision, "path": str(folder.resolve()), "sha256": hashes}
        if kind == "reward":
            if "score.weight" not in index["weight_map"]:
                raise ValueError("Reward classification head missing")
            with safe_open(folder / index["weight_map"]["score.weight"], framework="pt", device="cpu") as handle:
                head = handle.get_tensor("score.weight")
                if tuple(head.shape) != (1, 4096) or not torch.isfinite(head).all() or not torch.count_nonzero(head):
                    raise ValueError("Invalid reward classification head")
            lock["reward"]["head_shape"] = list(head.shape)
        print(f"Verified {kind}: {len(hashes)} file hashes", flush=True)
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    args.lock.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    validate_lock(args.lock, check_hashes=False)
    print(f"TPO artifacts PASS: {args.lock}", flush=True)


if __name__ == "__main__":
    main()
