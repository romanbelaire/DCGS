"""Offline, hash-pinned PEFT loading for native SafeDialBench generation."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_adapter_lock(path, base_model, base_revision):
    lock = json.loads(Path(path).read_text())
    if lock.get("schema_version") != 1:
        raise ValueError("Unsupported adapter lock schema")
    if (lock["base_model"], lock["base_revision"]) != (base_model, base_revision):
        raise ValueError("Adapter lock does not match requested base model/revision")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if sha256(Path(lock["adapter_path"]) / name) != lock["sha256"][name]:
            raise ValueError(f"Adapter hash mismatch: {name}")
    config = json.loads((Path(lock["adapter_path"]) / "adapter_config.json").read_text())
    if config["base_model_name_or_path"] != base_model:
        raise ValueError("Adapter declares a different base model")
    if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
        raise ValueError("Expected a causal-LM LoRA adapter")
    return lock


def attach_adapter(model, lock):
    from peft import PeftModel

    adapted = PeftModel.from_pretrained(
        model, lock["adapter_path"], is_trainable=False, local_files_only=True,
        autocast_adapter_dtype=False,
    )
    adapted.eval()
    adapted.requires_grad_(False)
    if not adapted.active_adapters:
        raise RuntimeError("No active PEFT adapter after loading")
    return adapted


def verify_adapter_effect(model, tokenizer, torch_module):
    """Fixed benign fixture, not an experimental control or safety judgment."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello. Please introduce yourself briefly."}],
        tokenize=False, add_generation_prompt=True,
    )
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    with torch_module.inference_mode():
        enabled = model(**encoded).logits[:, -1, :].float()
        with model.disable_adapter():
            disabled = model(**encoded).logits[:, -1, :].float()
    if not torch_module.isfinite(enabled).all() or not torch_module.isfinite(disabled).all():
        raise RuntimeError("Nonfinite logits during adapter verification")
    delta = float((enabled - disabled).abs().max().item())
    if delta <= 1e-6:
        raise RuntimeError("Adapter has no measurable effect on the fixed fixture")
    return {"passed": True, "max_abs_logit_difference": delta,
            "active_adapters": model.active_adapters, "merged": False}
