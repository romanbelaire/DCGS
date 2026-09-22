#!/usr/bin/env python3
"""SafeDial DCGS with a fixed 384-token belief-list budget and optional LL history."""
import argparse
import fcntl
import json
from pathlib import Path
from unittest.mock import patch

import run_safedial_dcgs_wildjailbreak as base
import safedial_dcgs_wildjailbreak as adapter
import validate_safedial_dcgs_wildjailbreak as auditor

PROTOCOL = "safedial_main_wildjailbreak_belief384_v1"


def configuration(method, device="cpu", artifacts=None, *, include_history=False):
    if type(include_history) is not bool:
        raise ValueError("include_history must be a boolean")
    # Check original policy invariants before applying the declared experiment settings.
    config = adapter.configuration(method, device, artifacts)
    config.ll_action_belief_only = not include_history
    config.freeform_max_new_tokens = 384
    return config


def method_spec(include_history):
    return {**adapter.METHOD_SPEC,
            "ll_input_context": "gold_history_and_selected_belief" if include_history else "selected_belief_only",
            "ll_history_policy": "all supplied prior gold turns and current user; no current reference answer",
            "actor_context_policy": "original tokenizer behavior and input-plus-output capacity check; no added trimming",
            "belief_generation_max_new_tokens": 384,
            "belief_budget_scope": "per generated candidate list",
            "experiment": PROTOCOL}


def source_hashes():
    # New files live outside src so active v3 runs keep their existing source identity.
    paths = [Path(__file__), adapter.ROOT / "scripts/slurm/run_safedial_dcgs_belief384_full.sbatch"]
    return {**base.source_hashes(), **{
        path.relative_to(adapter.ROOT).as_posix(): base.file_sha256(path) for path in paths}}


def manifest_for(args, selected, lock):
    manifest = base.manifest_for(args, selected, lock)
    config = configuration(args.method, args.device, lock, include_history=args.include_history)
    mode = "history" if args.include_history else "belief-only"
    manifest.update(protocol=PROTOCOL, include_history=args.include_history,
                    model_id=f"zephyr-7b-beta-{args.method}-main-belief384-{mode}-v1",
                    method_spec=method_spec(args.include_history), effective_config=vars(config),
                    source_sha256=source_hashes())
    return manifest


class LocalBackend(adapter.LocalBackend):
    def __init__(self, config, lock):
        super().__init__(config, lock)
        self.loading["method_spec"] = method_spec(not config.ll_action_belief_only)


def validate(folder, require_gpu=False, tokenizer=None):
    manifest = json.loads((Path(folder) / "run_config.json").read_text())
    include_history = manifest.get("include_history")
    if type(include_history) is not bool or manifest.get("protocol") != PROTOCOL:
        raise ValueError("Not a separately recorded 384-token belief experiment")
    expected_id = f"zephyr-7b-beta-{manifest['method']}-main-belief384-{'history' if include_history else 'belief-only'}-v1"
    if manifest.get("model_id") != expected_id:
        raise ValueError("Belief384 model identity mismatch")

    def configured(method, device="cpu", artifacts=None):
        return configuration(method, device, artifacts, include_history=include_history)

    # Reuse the complete journal/recovery/GPU auditor with this experiment's
    # identity. Patches are scoped to this synchronous audit, never saved to disk.
    with patch.object(auditor, "PROTOCOL", PROTOCOL), \
         patch.object(auditor, "METHOD_SPEC", method_spec(include_history)), \
         patch.object(auditor, "source_hashes", source_hashes), \
         patch.object(auditor, "configuration", configured):
        return auditor.validate(folder, require_gpu, tokenizer)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("vdcgs", "rdcgs"))
    parser.add_argument("--include-history", action="store_true",
                        help="Give the LL generator all supplied gold history plus the selected belief")
    parser.add_argument("--dataset", type=Path, default=base.DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--artifact-lock", type=Path, default=base.LOCK)
    parser.add_argument("--seed", type=int, default=0)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ids")
    selection.add_argument("--limit", type=int)
    selection.add_argument("--per-task", type=int)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--validate-only", action="store_true", help="Tokenizer-only preflight; no inference")
    actions.add_argument("--audit-only", action="store_true", help="Replay saved results using their recorded context mode")
    actions.add_argument("--recover-from", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--recovery-reason")
    parser.add_argument("--on-turn-error", choices=("stop", "record-and-continue"), default="stop")
    args = parser.parse_args(argv)
    if args.audit_only:
        if args.method or args.include_history or args.recovery_reason:
            parser.error("--audit-only reads method/context from the saved manifest; omit experiment flags")
        with (args.output_dir / ".lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            report = validate(args.output_dir, args.require_gpu)
            base.write_json(args.output_dir / "validation.json", report)
        print(json.dumps(report))
        return 0 if report["passed"] or args.allow_incomplete else 2
    if not args.method:
        parser.error("--method is required for generation/preflight/recovery")
    if args.require_gpu or args.allow_incomplete:
        parser.error("--require-gpu and --allow-incomplete require --audit-only")
    if args.recovery_reason and not args.recover_from:
        parser.error("--recovery-reason requires --recover-from")
    rows = base.load_jsonl(args.dataset)
    base.validate_dataset(rows, args.dataset)
    selected = base.select_dialogues(rows, args)
    lock = base.load_artifact_lock(args.artifact_lock)
    config = configuration(args.method, args.device, lock, include_history=args.include_history)
    manifest = json.loads(json.dumps(manifest_for(args, selected, lock)))
    # Reject cross-mode reuse before expensive artifact checks or model loading.
    path = args.output_dir / "run_config.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Manifest/source/context mismatch; use a fresh output directory")
    base.verify_artifacts(lock)
    tokenizer = base.tokenizer_for(lock)
    if args.recover_from:
        print(json.dumps(base.recover_run(args.recover_from, args.output_dir, manifest,
                         selected, config, tokenizer, args.recovery_reason)))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Manifest/source/context mismatch; use a fresh output directory")
        if not path.exists():
            base.write_json(path, manifest)
        if args.validate_only:
            report = base.preflight(selected, config, tokenizer)
            report["method_spec"] = method_spec(args.include_history)
            base.write_json(args.output_dir / "preflight.json", report)
            print(json.dumps(report))
            return 0
        return base.run_selected(args.output_dir, selected, manifest, config, tokenizer,
                                 lambda: LocalBackend(config, lock))


if __name__ == "__main__":
    raise SystemExit(main())
