#!/usr/bin/env python3
"""Stage the standard Qwen base and lock the user-supplied DCR adapter/tokenizer."""

import argparse
import json
from pathlib import Path

from safedial_adapters import sha256

BASE = "Qwen/Qwen2.5-1.5B"
BASE_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
BASE_WEIGHTS_SHA256 = "a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796"
BASE_FILES = ("config.json", "generation_config.json", "model.safetensors")
ADAPTER_FILES = (
    "adapter_config.json", "adapter_model.safetensors", "params.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja",
)
DEFAULT_LOCK = Path("configs/safedial/dcr.lock.json")


def inspect_adapter(adapter_path, base_config):
    import torch
    from safetensors import safe_open

    config = json.loads((adapter_path / "adapter_config.json").read_text())
    if (config.get("base_model_name_or_path"), config.get("peft_type"), config.get("task_type")) != (
        BASE, "LORA", "CAUSAL_LM"
    ):
        raise ValueError("DCR requires the standard Qwen2.5-1.5B causal-LM LoRA")
    if config.get("r") != 8 or set(config.get("target_modules", [])) != {"q_proj", "k_proj", "v_proj"}:
        raise ValueError("Unexpected DCR LoRA rank/targets")
    expected = {}
    hidden = base_config["hidden_size"]
    head_dim = base_config.get("head_dim", hidden // base_config["num_attention_heads"])
    for layer in range(base_config["num_hidden_layers"]):
        for projection in ("q_proj", "k_proj", "v_proj"):
            output = head_dim * base_config[
                "num_attention_heads" if projection == "q_proj" else "num_key_value_heads"
            ]
            prefix = f"base_model.model.model.layers.{layer}.self_attn.{projection}"
            expected[f"{prefix}.lora_A.weight"] = (8, hidden)
            expected[f"{prefix}.lora_B.weight"] = (output, 8)
    nonzero_b = 0
    with safe_open(adapter_path / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(expected):
            raise ValueError("DCR tensor names/layers do not match the base architecture")
        for key, shape in expected.items():
            tensor = handle.get_tensor(key)
            if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid DCR tensor shape or nonfinite weights: {key}")
            if "lora_B" in key and torch.count_nonzero(tensor).item():
                nonzero_b += 1
    if not nonzero_b:
        raise ValueError("DCR adapter has no nonzero LoRA B matrices")
    return {"tensor_count": len(expected), "nonzero_lora_b_count": nonzero_b}


def validate_lock_data(lock):
    if (lock.get("schema_version"), lock.get("method"), lock.get("base_model"),
            lock.get("base_revision")) != (1, "DCR", BASE, BASE_REVISION):
        raise ValueError("Unexpected DCR artifact lock identity")
    for root_key, hashes_key, required in (
        ("base_path", "base_sha256", BASE_FILES),
        ("adapter_path", "sha256", ADAPTER_FILES),
    ):
        hashes = lock[hashes_key]
        if set(hashes) != set(required):
            raise ValueError(f"Incomplete DCR hash inventory: {hashes_key}")
        root = Path(lock[root_key])
        # Fail if an unpinned config could redirect base loading to an adapter.
        if root_key == "base_path" and (root / "adapter_config.json").exists():
            raise ValueError("Base snapshot unexpectedly contains a PEFT adapter")
        for name in required:
            if sha256(root / name) != hashes[name]:
                raise ValueError(f"DCR artifact hash mismatch: {name}")
    if lock["base_sha256"]["model.safetensors"] != BASE_WEIGHTS_SHA256:
        raise ValueError("Base weights differ from official pinned Qwen LFS hash")
    config = json.loads((Path(lock["adapter_path"]) / "adapter_config.json").read_text())
    if config.get("base_model_name_or_path") != BASE:
        raise ValueError("Adapter declares a different base")
    return lock


def validate_lock(path):
    return validate_lock_data(json.loads(Path(path).read_text()))


def load_tokenizer(lock):
    from transformers import AutoTokenizer

    path = Path(lock["adapter_path"])
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    # Explicit assignment also supports Transformers versions that don't discover .jinja.
    tokenizer.chat_template = (path / "chat_template.jinja").read_text()
    if not tokenizer.chat_template.strip() or tokenizer.eos_token_id is None:
        raise ValueError("DCR tokenizer must define its training template and EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use cached base files only.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface/hub"))
    parser.add_argument("--adapter-path", type=Path, default=Path("models/dcr_sft_lora_171151"))
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    if args.lock.exists():
        lock = validate_lock(args.lock)
        if Path(lock["adapter_path"]).resolve() != args.adapter_path.resolve():
            raise ValueError("Existing lock refers to a different adapter directory")
    else:
        from huggingface_hub import snapshot_download

        base_path = Path(snapshot_download(
            BASE, revision=BASE_REVISION, cache_dir=str(args.cache_dir.resolve()),
            allow_patterns=list(BASE_FILES), local_files_only=args.offline, token=False,
        ))
        adapter_path = args.adapter_path.resolve()
        lock = {
            "schema_version": 1, "method": "DCR", "base_model": BASE,
            "base_revision": BASE_REVISION, "base_path": str(base_path.resolve()),
            "adapter_repo": "local/dcr_sft_lora_171151", "adapter_path": str(adapter_path),
            "base_sha256": {name: sha256(base_path / name) for name in BASE_FILES},
            "sha256": {name: sha256(adapter_path / name) for name in ADAPTER_FILES},
            "provenance": {
                "base": "User confirmed standard Qwen/Qwen2.5-1.5B; no separate refined base.",
                "template": "User confirmed uploaded chat_template.jinja was used for training.",
                "revision": "Training base revision unspecified; inference pinned to public snapshot.",
            },
        }
        validate_lock_data(lock)
    config = json.loads((Path(lock["base_path"]) / "config.json").read_text())
    report = inspect_adapter(Path(lock["adapter_path"]), config)
    tokenizer = load_tokenizer(lock)
    if max(tokenizer.get_vocab().values()) >= config["vocab_size"]:
        raise ValueError("Uploaded tokenizer requires embeddings absent from the standard base")
    fixture = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello."}], tokenize=False, add_generation_prompt=True,
    )
    if fixture != "USER: Hello.\nASSISTANT: ":
        raise ValueError("Unexpected DCR training template rendering")
    if not args.lock.exists():
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        with args.lock.open("x") as handle:
            handle.write(json.dumps(lock, indent=2) + "\n")
    print(json.dumps({"passed": True, "lock": str(args.lock), "base_revision": BASE_REVISION,
                      **report, "fixture": fixture}, indent=2))


if __name__ == "__main__":
    main()
