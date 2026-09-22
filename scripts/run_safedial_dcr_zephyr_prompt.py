#!/usr/bin/env python3
"""DCR diagnostic with literal Zephyr chat formatting and the Qwen tokenizer.

The earlier training/raw diagnostic is loaded into a private module namespace.
Only this instance's formatting hooks change; previously offered runners and
their source hashes remain untouched, including when jobs are already running.
"""

import importlib.util
from pathlib import Path
import re
import sys

import run_safedial_dcr as original

FORMAT = "zephyr"
PROTOCOL = "safedial_dcr_prompt_ablation_zephyr_literal_v1"
MODEL_ID = "qwen2.5-1.5b-dcr-sft-171151-prompt-zephyr-literal-v1"


def identities(prompt_format):
    if prompt_format != FORMAT:
        raise ValueError("This runner only accepts the literal Zephyr prompt format")
    return PROTOCOL, MODEL_ID


def prompt_policy(prompt_format):
    identities(prompt_format)
    return {"tokenizer": "uploaded_adapter", "prompt_format": FORMAT,
            "template": "literal_zephyr_role_markers_v1",
            "serialization": "role_marker_newline_content_literal_end_s_newline",
            "assistant_cue": "<|assistant|>\n", "message_ending": "</s>\n",
            "system_message": None, "truncation": "reject_overflow", "add_special_tokens": False,
            "stop_policy": "uploaded_Qwen_EOS_only; literal_</s>_is_not_a_stop_sequence",
            "vocabulary_changes": False}


def render_prompt(tokenizer, history, index, prompt_format):
    identities(prompt_format)
    messages = original.baseline.gold_messages(history, index)
    rendered = "".join(f"<|{m['role']}|>\n{m['content']}</s>\n" for m in messages)
    return messages, rendered + "<|assistant|>\n"


def marker_diagnostics(tokenizer):
    return {"markers": {
        marker: {"token_ids": tokenizer(marker, add_special_tokens=False)["input_ids"],
                 "registered_special_token": marker in tokenizer.all_special_tokens}
        for marker in ("<|user|>", "<|assistant|>", "</s>")
    }, "generation_eos_token": tokenizer.eos_token,
       "generation_eos_token_id": tokenizer.eos_token_id,
       "added_stop_sequences": []}


def parse_args(argv=None):
    args = original.parse_args(argv)
    if args.output_dir == Path("outputs/safedial_baseline/dcr_qwen_1.5b_v1"):
        args.output_dir = Path("outputs/safedial_baseline/dcr_qwen_1.5b_prompt_zephyr_ids1-2-3_smoke_v1")
    if args.ids is None and args.per_task is None and args.limit is None:
        args.ids = "1,2,3"
    args.prompt_format = FORMAT
    args.protocol, args.model_id = identities(FORMAT)
    return args


def build_runtime():
    path = Path(__file__).resolve().with_name("run_safedial_dcr_prompt_ablation.py")
    spec = importlib.util.spec_from_file_location("_dcr_zephyr_prompt_runtime", path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.SOURCES = (*runtime.SOURCES, "run_safedial_dcr.py", Path(__file__).name)
    runtime.identities = identities
    runtime.prompt_policy = prompt_policy
    runtime.render_prompt = render_prompt
    runtime.parse_args = parse_args
    original_preflight = runtime.preflight
    original_audit = runtime.audit_run

    def preflight(selected, tokenizer, config, max_new_tokens, prompt_format):
        result = original_preflight(selected, tokenizer, config, max_new_tokens, prompt_format)
        return {**result, "marker_diagnostics": marker_diagnostics(tokenizer)}

    def audit_run(output_dir, require_gpu=False):
        report = original_audit(output_dir, require_gpu)
        records = runtime.baseline.load_jsonl(Path(output_dir) / "turns.jsonl")
        for metrics, record in zip(report["turn_metrics"], records):
            metrics["generated_zephyr_role_markers"] = len(re.findall(
                r"<\|(?:user|assistant|system)\|>", record["generated_response"]))
            metrics["generated_literal_end_s"] = record["generated_response"].count("</s>")
        return report

    runtime.preflight = preflight
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
