#!/usr/bin/env python3
"""Validate complete native coverage, both selection stages, journals and GPU load."""
import argparse
import json
from pathlib import Path

import run_safedial_baseline as base
from run_safedial_dcgs_methods import implementation_hashes
from safedial_dcgs_methods import defense_config, validate_audit
from validate_safedial_generation import validate_run


def validate_dcgs(folder, require_gpu=False):
    folder = Path(folder)
    report = validate_run(folder)
    manifest = json.loads((folder / "run_config.json").read_text())
    if manifest["defense"] != defense_config(manifest["defense"]["name"]) or manifest["implementation_sha256"] != implementation_hashes():
        raise ValueError("DCGS policy or source hashes changed")
    if manifest["num_choices"] != 1:
        raise ValueError("Expected one evaluated response per turn")
    for key in ("high_level", "token_critic"):
        artifact = manifest["artifacts"][key]
        if base.file_sha256(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError(f"DCGS {key} artifact changed")
    run_id = base.stable_id(json.dumps(manifest, sort_keys=True))
    journal = {}
    for entry in base.load_jsonl(folder / "events.jsonl"):
        if entry["run_id"] != run_id:
            raise ValueError("Foreign DCGS event")
        journal[(entry["dialogue_id"], entry["turn_index"], entry["event"]["index"])] = entry["event"]
    used = set()
    counts = {"generate": 0, "hl_score": 0, "ll_score": 0}
    capped = 0
    for record in base.load_jsonl(folder / "turns.jsonl"):
        if (record["run_id"] != run_id or record["seed"] != base.turn_seed(manifest["seed"], manifest["model_id"], record["dialogue_id"], 0, record["turn_index"])
                or record["answer_id"] != base.stable_id(run_id, record["dialogue_id"])):
            raise ValueError("DCGS turn identity or seed mismatch")
        validate_audit(record, manifest["defense"])
        for event in record["dcgs"]["events"]:
            key = (record["dialogue_id"], record["turn_index"], event["index"])
            if journal.get(key) != event:
                raise ValueError("DCGS journal/turn mismatch")
            used.add(key)
            counts[event["request"]["kind"]] += 1
            capped += bool(event["result"].get("hit_token_cap"))
    if used != set(journal):
        raise ValueError("Unexpected DCGS events")
    if require_gpu:
        if not all((folder / n).is_file() for n in ("model_loading.json", "runtime_stats.json", "context_probe.json")):
            raise ValueError("GPU model-loading/runtime evidence missing")
        probe = json.loads((folder / "context_probe.json").read_text())
        preflight = json.loads((folder / "preflight.json").read_text())
        if probe.get("passed") is not True or probe.get("sequence_tokens") != preflight["startup_probe_tokens"]:
            raise ValueError("Successful context capacity probe missing")
        loading = json.loads((folder / "model_loading.json").read_text())
        runtime = json.loads((folder / "runtime_stats.json").read_text())
        if not runtime.get("peak_gpu_allocated_bytes") or runtime.get("exit_code") != 0 or not runtime.get("cuda_ready"):
            raise ValueError("Successful GPU runtime/memory evidence missing")
        if set(loading) != {"actor_and_hl", "ll_encoder", "heads"}:
            raise ValueError("Both critic backbones and heads must be loaded")
        if any(not x.get("passed") or x.get("trainable_parameters") != 0 for x in loading.values()):
            raise ValueError("Frozen checkpoint loading not verified")
        if loading["actor_and_hl"]["dtype"] != "torch.bfloat16" or loading["ll_encoder"]["dtype"] != "torch.float16":
            raise ValueError("Incorrect critic backbone precision")
        devices = {"actor_and_hl": manifest["device"], "ll_encoder": manifest["ll_device"]}
        if any(loading[name].get("device") != device for name, device in devices.items()):
            raise ValueError("Loaded encoder placement differs from manifest")
        if set(runtime.get("gpu_devices", {})) != set(devices.values()):
            raise ValueError("Per-GPU memory evidence missing")
        if any(not item.get("peak_allocated_bytes") for item in runtime["gpu_devices"].values()):
            raise ValueError("GPU allocation evidence missing")
        expected_heads = ["q_mlp_head_state_dict", "harm_head", "follow_head"]
        if manifest["defense"]["use_regret"]:
            expected_heads.append("regret_mlp_head_state_dict")
        if loading["heads"].get("required_heads_strictly_loaded") != sorted(expected_heads):
            raise ValueError("Incomplete critic head loading")
        head_devices = {name: manifest["ll_device"] if name in ("harm_head", "follow_head")
                        else manifest["device"] for name in expected_heads}
        if loading["heads"].get("devices") != head_devices:
            raise ValueError("Loaded head placement differs from manifest")
        for kind, key in (("high_level", "high_level_checkpoint_sha256"), ("token_critic", "token_checkpoint_sha256")):
            if loading["heads"].get(key) != manifest["artifacts"][kind]["sha256"]:
                raise ValueError("Loaded checkpoint differs from manifest")
    report.update(dcgs_audit_passed=True, call_counts=counts, generations_at_token_cap=capped,
                  response_candidates=report["turns"] * 5, input_truncated_turns=0)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    report = validate_dcgs(args.output_dir, args.require_gpu)
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
