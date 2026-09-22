#!/usr/bin/env python3
"""Paired DCR smoke with the pinned base Qwen tokenizer's standard template."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import run_safedial_dcr as original

FORMAT = "qwen_standard"
PROTOCOL = "safedial_dcr_prompt_ablation_qwen_standard_v1"
MODEL_ID = "qwen2.5-1.5b-dcr-sft-171151-prompt-qwen-standard-v1"
ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_CONFIG = ROOT / "configs/safedial/qwen_standard_template/tokenizer_config.json"
CONFIG_SHA256 = "c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9"
SYSTEM_MESSAGE = "You are a helpful assistant."


def standard_template():
    data = TEMPLATE_CONFIG.read_bytes()
    if hashlib.sha256(data).hexdigest() != CONFIG_SHA256:
        raise ValueError("Pinned Qwen tokenizer configuration hash mismatch")
    return json.loads(data)["chat_template"]


def identities(prompt_format):
    if prompt_format != FORMAT:
        raise ValueError("This runner only accepts the standard Qwen prompt format")
    return PROTOCOL, MODEL_ID


def prompt_policy(prompt_format):
    identities(prompt_format)
    return {"tokenizer": "uploaded_adapter", "prompt_format": FORMAT,
            "template": "official_base_Qwen_tokenizer_config.chat_template",
            "template_repository": original.BASE, "template_revision": original.BASE_REVISION,
            "tokenizer_config_sha256": CONFIG_SHA256,
            "template_sha256": hashlib.sha256(standard_template().encode()).hexdigest(),
            "system_message": SYSTEM_MESSAGE, "system_message_source": "template_default",
            "assistant_cue": "<|im_start|>assistant\n", "message_ending": "<|im_end|>\n",
            "truncation": "reject_overflow", "add_special_tokens": False,
            "stop_policy": "uploaded_Qwen_EOS_only; no_added_im_end_stop",
            "vocabulary_changes": False}


def render_prompt(tokenizer, history, index, prompt_format):
    identities(prompt_format)
    messages = original.baseline.gold_messages(history, index)
    rendered = tokenizer.apply_chat_template(messages, chat_template=standard_template(),
                                             tokenize=False, add_generation_prompt=True)
    return messages, rendered


def marker_diagnostics(tokenizer):
    return {"markers": {
        marker: {"token_ids": tokenizer(marker, add_special_tokens=False)["input_ids"],
                 "registered_special_token": marker in tokenizer.all_special_tokens}
        for marker in ("<|im_start|>", "<|im_end|>")
    }, "generation_eos_token": tokenizer.eos_token,
       "generation_eos_token_id": tokenizer.eos_token_id, "added_stop_sequences": []}


def parse_args(argv=None):
    args = original.parse_args(argv)
    if args.output_dir == Path("outputs/safedial_baseline/dcr_qwen_1.5b_v1"):
        args.output_dir = Path("outputs/safedial_baseline/dcr_qwen_1.5b_prompt_qwen_standard_ids1-2-3_smoke_v1")
    if args.ids is None and args.per_task is None and args.limit is None:
        args.ids = "1,2,3"
    args.prompt_format = FORMAT
    args.protocol, args.model_id = identities(FORMAT)
    return args


def build_runtime():
    path = Path(__file__).resolve().with_name("run_safedial_dcr_prompt_ablation.py")
    spec = importlib.util.spec_from_file_location("_dcr_qwen_prompt_runtime", path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.SOURCES = (*runtime.SOURCES, "run_safedial_dcr.py", Path(__file__).name)
    runtime.identities = identities
    runtime.prompt_policy = prompt_policy
    runtime.render_prompt = render_prompt
    runtime.parse_args = parse_args
    original_preflight, original_audit = runtime.preflight, runtime.audit_run
    original_generate = runtime.generate_rendered

    def preflight(selected, tokenizer, config, max_new_tokens, prompt_format):
        result = original_preflight(selected, tokenizer, config, max_new_tokens, prompt_format)
        return {**result, "system_message": SYSTEM_MESSAGE,
                "marker_diagnostics": marker_diagnostics(tokenizer),
                "prompt_policy": prompt_policy(prompt_format)}

    def generate_rendered(**kwargs):
        result = original_generate(**kwargs)
        # Retain actual IDs even if the inherited runner rejects decoded-empty output.
        runtime.baseline.append_jsonl(kwargs["args"].output_dir / "raw_generations.jsonl", [{
            "seed": kwargs["seed"], "rendered_prompt_sha256": runtime.text_hash(kwargs["rendered"]),
            **result,
        }])
        return result

    def audit_run(output_dir, require_gpu=False):
        report = original_audit(output_dir, require_gpu)
        manifest = json.loads((Path(output_dir) / "run_config.json").read_text())
        tokenizer = runtime.load_tokenizer(manifest["adapter"])
        marker_ids = {marker: tokenizer.convert_tokens_to_ids(marker)
                      for marker in ("<|im_start|>", "<|im_end|>")}
        records = runtime.baseline.load_jsonl(Path(output_dir) / "turns.jsonl")
        for metrics, record in zip(report["turn_metrics"], records):
            # These special tokens are removed by skip_special_tokens=True.
            metrics["generated_qwen_marker_counts"] = {
                marker: record["completion_token_ids"].count(token_id)
                for marker, token_id in marker_ids.items()}
        return report

    runtime.preflight = preflight
    runtime.generate_rendered = generate_rendered
    runtime.audit_run = audit_run
    return runtime


def main(argv=None):
    return build_runtime().main(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
