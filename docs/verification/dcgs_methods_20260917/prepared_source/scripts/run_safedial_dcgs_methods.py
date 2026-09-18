#!/usr/bin/env python3
"""Separate SafeDialBench VDCGS/RDCGS runs; defaults to the full dataset.

All events resume from a durable journal. Full generation setup is gated on a
successful GPU smoke. No paid judge calls or Slurm submission occurs here.
"""
import argparse
import contextlib
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import run_safedial_baseline as base
from safedial_dcgs_methods import DCGSFailure, defense_config, generate_dcgs, validate_audit
from safedial_dcgs_methods_runtime import DEFAULT_LOCK, LocalBackend, validate_lock, tokenizer_and_preflight

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('vdcgs','rdcgs'), required=True)
    parser.add_argument('--dataset', type=Path, default=base.DEFAULT_DATASET)
    parser.add_argument('--artifact-lock', type=Path, default=DEFAULT_LOCK)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--model-id', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--ids')
    selection.add_argument('--per-task', type=int)
    selection.add_argument('--limit', type=int)
    parser.add_argument('--retry-errors', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.set_defaults(model='HuggingFaceH4/zephyr-7b-beta', revision='892b3d7a7b1cf10c7a701c60881cd93df615734c',
        temperature=0.7, top_p=1.0, max_new_tokens=1024, num_choices=1, dtype='bfloat16', max_input_tokens=None)
    args=parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir=Path(f'outputs/safedial_dcgs/zephyr_{args.method}_aug11_full')
    if args.model_id is None:
        args.model_id=f'zephyr-7b-beta-{args.method}-wildjailbreak-aug11-v2'
    base.validate_args(args)
    return args


def implementation_hashes():
    files=list((ROOT/'src').rglob('*.py')) + list((ROOT/'src/prompts').glob('*.jsonl'))
    files += [ROOT/'scripts'/name for name in ('run_safedial_baseline.py', 'run_safedial_dcgs.py',
        'run_safedial_dcgs_methods.py','safedial_dcgs_methods.py','safedial_dcgs_methods_runtime.py','safedial_dcgs_native_context.py','validate_safedial_dcgs_methods.py')]
    return {str(p.relative_to(ROOT)):base.file_sha256(p) for p in sorted(files)}


def manifest_for(args, selected, lock):
    manifest=base.manifest_for(args,args.dataset,selected)
    manifest.update(defense=defense_config(args.method), artifacts=lock, implementation_sha256=implementation_hashes(),
        backend='frozen_bf16_actor_hl_and_fp16_ll_encoder_serial_sdpa',
        resume='per_event_and_turn; failed_calls_retained; complete_dialogues_only_in_answers',
        token_accounting='all_generation_tokens; critic_context_tokens_separate', device=args.device)
    return manifest


@contextlib.contextmanager
def single_writer(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".generation.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another DCGS writer is active") from exc
        yield


def atomic_jsonl(path, records):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_journal(path):
    """Recover only an interrupted final append, retaining its exact bytes."""
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        end = data.rfind(b"\n") + 1
        tail = data[end:]
        try:
            json.loads(tail)
        except (ValueError, UnicodeDecodeError):
            recovery = path.parent / "recovery"
            recovery.mkdir(exist_ok=True)
            (recovery / f"{path.name}.{time.time_ns()}.partial").write_bytes(tail)
            with path.open("r+b") as handle:
                handle.truncate(end)
                handle.flush()
                os.fsync(handle.fileno())
            print(f"Recovered interrupted final append in {path}; original tail retained", flush=True)
        else:
            with path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
    return base.load_jsonl(path)


def load_state(args, manifest, selected):
    run_id = base.stable_id(json.dumps(manifest, sort_keys=True))
    valid_keys = {(r["id"], i) for r in selected for i in range(len(r["history"]))}
    turns, events = {}, {}
    for record in read_journal(args.output_dir / "turns.jsonl"):
        key = (record["dialogue_id"], record["turn_index"])
        if record["run_id"] != run_id or key not in valid_keys or record["model_id"] != args.model_id:
            raise ValueError("Foreign turn record in DCGS output")
        if record.get("choice_index") != 0 or record.get("answer_id") != base.stable_id(run_id, key[0]):
            raise ValueError("Incorrect saved DCGS turn identity")
        turns[key] = record
    for row in read_journal(args.output_dir / "events.jsonl"):
        key = (row["dialogue_id"], row["turn_index"])
        if row["run_id"] != run_id or key not in valid_keys:
            raise ValueError("Foreign event record in DCGS output")
        if type(row["event"]["index"]) is not int or row["event"]["index"] < 0:
            raise ValueError("Invalid DCGS event index")
        events[(*key, row["event"]["index"])] = row["event"]
    by_turn = {}
    for dialogue, turn, index in events:
        by_turn.setdefault((dialogue, turn), []).append(index)
    for indices in by_turn.values():
        if sorted(indices) != list(range(len(indices))):
            raise ValueError("Noncontiguous DCGS event journal")
    return run_id, turns, events


def export_answers(args, selected, turns, run_id):
    answers = []
    for row in selected:
        records = [turns.get((row["id"], i)) for i in range(len(row["history"]))]
        if any(r is None or r.get("error") for r in records):
            continue
        answers.append({"id": row["id"], "task": row["task"], "method": row["method"],
                        "model_id": args.model_id, "answer_id": base.stable_id(run_id, row["id"]),
                        "choices": [{"index": 0, "turns": [{"role": "assistant", "message": r["generated_response"]} for r in records]}],
                        "tstamp": records[-1]["tstamp"]})
    atomic_jsonl(args.output_dir / "answers.jsonl", answers)


def generate_selected(args, manifest, selected, execute):
    run_id, turns, events = load_state(args, manifest, selected)
    # Refuse to reuse altered success records, including altered gold histories.
    for row in selected:
        for index in range(len(row["history"])):
            previous = turns.get((row["id"], index))
            if previous and not previous.get("error"):
                if previous["prompt_history"] != base.gold_messages(row["history"], index):
                    raise ValueError("Saved DCGS gold history changed")
                if previous["seed"] != base.turn_seed(args.seed, args.model_id, row["id"], 0, index):
                    raise ValueError("Saved DCGS seed changed")
                validate_audit(previous, manifest["defense"])
                saved = [events[k] for k in sorted(events) if k[:2] == (row["id"], index)]
                if saved != previous["dcgs"]["events"]:
                    raise ValueError("Saved DCGS event journal differs from completed turn audit")
    export_answers(args, selected, turns, run_id)
    for row in selected:
        for index, source in enumerate(row["history"]):
            key = (row["id"], index)
            previous = turns.get(key)
            if previous and not previous.get("error"):
                continue
            if previous and previous.get("error") and not args.retry_errors:
                print(f"Unresolved DCGS error {key}; inspect audit before --retry-errors", flush=True)
                return 2
            cursor = 0

            def cached_execute(request):
                nonlocal cursor
                saved = events.get((*key, cursor))
                cursor += 1
                if saved and saved["request"] != request:
                    raise ValueError("Saved DCGS event request changed")
                if saved and not saved.get("error"):
                    return saved["result"]
                return execute(request)

            def persist(event):
                event_key = (*key, event["index"])
                if events.get(event_key) == event:
                    return
                base.append_jsonl(args.output_dir / "events.jsonl", [{"run_id": run_id, "dialogue_id": key[0],
                                                                    "turn_index": key[1], "event": event}])
                events[event_key] = event
                if event.get("format_warnings"):
                    print(f"WARNING dialogue={key[0]} turn={key[1]} event={event['index']}: "
                          + "; ".join(event["format_warnings"]), flush=True)

            messages = base.gold_messages(row["history"], index)
            seed = base.turn_seed(args.seed, args.model_id, row["id"], 0, index)
            record = {"run_id": run_id, "answer_id": base.stable_id(run_id, row["id"]),
                      "benchmark": "SafeDialBench", "protocol": base.PROTOCOL,
                      "model": args.model, "model_id": args.model_id, "dialogue_id": row["id"],
                      "task": row["task"], "method": row["method"], "scene": row["scene"],
                      "dataset_model_type": row.get("model_type"), "choice_index": 0, "turn_index": index,
                      "seed": seed, "prompt_history": messages, "user_message": source["user"],
                      "reference_response": source["bot"], "error": None}
            try:
                result = generate_dcgs(messages, manifest["defense"], seed, cached_execute, persist)
                record.update({k: v for k, v in result.items() if k != "message"}, generated_response=result["message"])
            except DCGSFailure as exc:
                record.update(error=str(exc), generated_response=None, dcgs=exc.audit)
            record["tstamp"] = time.time()
            base.append_jsonl(args.output_dir / "turns.jsonl", [record])
            turns[key] = record
            if record["error"]:
                export_answers(args, selected, turns, run_id)
                print(f"FAIL dialogue={key[0]} turn={key[1]}: {record['error']}", flush=True)
                return 2
            print(f"Saved dialogue={key[0]} turn={key[1]} candidates={len(record['dcgs']['responses'])}", flush=True)
        export_answers(args, selected, turns, run_id)
    atomic_jsonl(args.output_dir / "turns.jsonl", [turns[key] for key in sorted(turns)])
    return 0


def main(argv=None):
    args=parse_args(argv)
    rows=base.load_jsonl(args.dataset)
    base.validate_dataset(rows,args.dataset)
    selected=base.select_dialogues(rows,args)
    if not selected or len({r['id'] for r in selected})!=len(selected):
        raise ValueError('Selection must be nonempty and unique')
    print('Checking pinned Zephyr and both WildJailbreak critic checkpoints',flush=True)
    lock=validate_lock(args.artifact_lock, args.method)
    tokenizer,actor_limit,preflight=tokenizer_and_preflight(lock,selected,args.method)
    print(json.dumps(preflight),flush=True)
    if args.validate_only:
        return 0
    manifest=manifest_for(args,selected,lock)
    with single_writer(args.output_dir):
        if not (args.output_dir/'run_config.json').exists() and any((args.output_dir/n).exists() for n in ('answers.jsonl','turns.jsonl','events.jsonl')):
            raise ValueError('Existing DCGS output lacks a manifest')
        base.ensure_manifest(args.output_dir/'run_config.json',manifest)
        (args.output_dir/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
        backend=None
        started=time.perf_counter()
        calls={'generate':0,'hl_score':0,'ll_score':0}
        status=2
        failure=None
        gpu_ready=False
        import torch
        def execute(request):
            nonlocal backend,gpu_ready
            if backend is None:
                if args.device.startswith('cuda'):
                    if not torch.cuda.is_available():
                        raise RuntimeError('CUDA unavailable; user must submit GPU smoke')
                    torch.cuda.init()
                    torch.cuda.set_device(torch.device(args.device))
                    torch.cuda.reset_peak_memory_stats(args.device)
                    gpu_ready=True
                backend=LocalBackend(args,lock,tokenizer,actor_limit)
                (args.output_dir/'model_loading.json').write_text(json.dumps(backend.loading,indent=2)+'\n')
                probe={'passed':False,'sequence_tokens':preflight['startup_probe_tokens']}
                try:
                    probe=backend.probe_context(preflight['startup_probe_tokens'])
                except Exception as exc:
                    probe['error']=f'{type(exc).__name__}: {exc}'
                    raise
                finally:
                    (args.output_dir/'context_probe.json').write_text(json.dumps(probe,indent=2)+'\n')
                print(f"Context capacity probe PASS: {preflight['startup_probe_tokens']} tokens",flush=True)
            calls[request['kind']]+=1
            return backend(request)
        try:
            status=generate_selected(args,manifest,selected,execute)
            return status
        except Exception as exc:
            failure=f'{type(exc).__name__}: {exc}'
            raise
        finally:
            stats={'elapsed_seconds':time.perf_counter()-started,'device':args.device,'exit_code':status,
                'failure':failure,'actual_calls':calls,'model_loading_performed':backend is not None,
                'cuda_ready':gpu_ready,'torch_version':torch.__version__,
                'gpu_name':torch.cuda.get_device_name(args.device) if gpu_ready else None,
                'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(args.device) if gpu_ready else None,
                'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(args.device) if gpu_ready else None,
                'scope':'current_invocation_includes_both_backbones_loading_and_synthetic_context_probe; excludes_artifact_preflight'}
            if any(calls.values()) or not (args.output_dir/'runtime_stats.json').exists():
                (args.output_dir/'runtime_stats.json').write_text(json.dumps(stats,indent=2)+'\n')
            base.append_jsonl(args.output_dir/'invocations.jsonl',[stats])


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (FileNotFoundError,RuntimeError,ValueError) as exc:
        print(f'FAIL: {exc}',file=sys.stderr)
        raise SystemExit(2)
