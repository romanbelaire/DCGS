#!/usr/bin/env python3
"""Stage pinned official CAT weights; no model generation or paid API calls."""

import argparse
import json
from pathlib import Path

from safedial_adapters import sha256, validate_adapter_lock


BASE = "HuggingFaceH4/zephyr-7b-beta"
BASE_REVISION = "892b3d7a7b1cf10c7a701c60881cd93df615734c"
ADAPTER = "ContinuousAT/Zephyr-CAT"
ADAPTER_REVISION = "550ea10d3d0f867f62e205d928029573e0575e1b"
WEIGHTS_SHA256 = "48119f54d8a2db7e6613c28a3525d83d40fdb8c8805b66dfd8086e04ca9d51da"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface/hub"))
    parser.add_argument("--lock", type=Path, default=Path("configs/safedial/cat.lock.json"))
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    import torch

    # Reuse the existing base. Never download another 7B model implicitly.
    base_path = Path(snapshot_download(
        BASE, revision=BASE_REVISION, cache_dir=str(args.cache_dir.resolve()),
        local_files_only=True,
    ))
    index = json.loads((base_path / "model.safetensors.index.json").read_text())
    for filename in set(index["weight_map"].values()) | {"config.json", "tokenizer_config.json"}:
        if not (base_path / filename).is_file():
            raise FileNotFoundError(base_path / filename)
    adapter_path = Path(snapshot_download(
        ADAPTER, revision=ADAPTER_REVISION, cache_dir=str(args.cache_dir.resolve()),
        allow_patterns=["adapter_config.json", "adapter_model.safetensors", "README.md"],
        local_files_only=args.offline, token=False,
    ))
    hashes = {name: sha256(adapter_path / name)
              for name in ("adapter_config.json", "adapter_model.safetensors", "README.md")}
    if hashes["adapter_model.safetensors"] != WEIGHTS_SHA256:
        raise ValueError("CAT weights do not match official LFS SHA-256")
    torch.set_num_threads(2)
    tensor_count = 0
    nonzero_b = 0
    with safe_open(adapter_path / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite adapter tensor: {key}")
            tensor_count += 1
            if "lora_B" in key and torch.count_nonzero(tensor).item():
                nonzero_b += 1
    if not nonzero_b:
        raise ValueError("Adapter contains no nonzero LoRA B matrices")
    lock = {
        "schema_version": 1, "method": "CAT", "base_model": BASE,
        "base_revision": BASE_REVISION, "base_path": str(base_path.resolve()),
        "adapter_repo": ADAPTER, "adapter_revision": ADAPTER_REVISION,
        "adapter_path": str(adapter_path.resolve()), "sha256": hashes,
        "adapter_tensor_count": tensor_count, "nonzero_lora_b_count": nonzero_b,
        "base_revision_note": "Adapter config leaves base revision unspecified; pinned to existing local Zephyr snapshot.",
        "license_note": "Adapter model card does not declare a license; retain source README and review upstream terms before redistribution.",
    }
    if args.lock.exists():
        if json.loads(args.lock.read_text()) != lock:
            raise ValueError("Existing CAT lock differs; refusing to overwrite")
    else:
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        args.lock.write_text(json.dumps(lock, indent=2) + "\n")
    validate_adapter_lock(args.lock, BASE, BASE_REVISION)
    print(f"CAT artifacts validated: {tensor_count} tensors, {nonzero_b} nonzero LoRA B matrices")
    print(f"Lock: {args.lock}; adapter revision: {ADAPTER_REVISION}")


if __name__ == "__main__":
    main()
