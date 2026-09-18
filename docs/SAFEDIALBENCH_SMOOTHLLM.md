# SafeDialBench SmoothLLM

> Script cleanup (2026-09-18): superseded launchers and one-time migration
> scripts referenced below are now in the [audit archive](../archive/scripts_cleanup_20260918/README.md).
> Use the [current script index](../scripts/README.md) for active entry points.
> The original text below is retained as experiment history.

**2026-09-15:** the next-allocation [turn-level continuation](SAFEDIALBENCH_SMOOTHLLM_TURN_RESUME.md)
is implemented and tested. It saves each turn and retries only failed/missing
turns. Current job 252143 and the historical launcher below still use
dialogue-level resume. Use the new launcher after that job ends.

> GPU request update (2026-09-10): the full launcher now requests one L40
> (48 GB) instead of A40, with the same 48-hour limit and inference settings.
> A40 smoke measurements below remain historical; full L40 timing is unmeasured.
> Editing the file does not update an already queued job. User must run
> `scontrol update JobId=245412 Gres=gpu:l40:1`
> while it is pending, then verify the request. No agent scheduler mutation.

Status: GPU smoke 245317 passed generation, all 40 candidate audits and judge
dry-run. Safety judging is pending. Full resumable launcher prepared following
the 2026-09-10 request to proceed directly to full generation. CAT job 245229
remains pending and unchanged.

## Frozen smoke setup

- Unadapted `HuggingFaceH4/zephyr-7b-beta`, revision
  `892b3d7a7b1cf10c7a701c60881cd93df615734c`, cached locally. No new weights.
- Eight copies, random character substitution at 10%, `string.printable`;
  replacement count is floor(length * 10 / 100). Replacement can equal the
  original character; empty/tiny spans may remain unchanged.
- Only the current user message is perturbed, before the native chat template.
  All prior user/gold-assistant messages are restored for every benchmark turn.
  No generated-history replay, extra system prompt, or `[RESPONSE]` prefill.
- Greedy candidate decoding, 1,024 new tokens per copy, seed 0, BF16, sequential
  batch size 1. Greedy differs from the plain/CAT sampling temperature 0.7.
- Case-sensitive refusal-substring detector, strict >50% jailbreak vote,
  refusal-side ties, uniform random selection of an indexed majority candidate.
  Duplicate responses retain separate votes. This heuristic is not measured ASR.

The algorithm reference is the [official implementation at commit 1855c87](https://github.com/arobey1/smooth-llm/tree/1855c8791d4ffbcd902abcdd1b5ef69fda1a96e0/lib).
Detector, swap, vote and selection semantics were checked against it. Eight
copies/10%, Zephyr's chat template, explicit greedy/BF16/1,024-token decoding and
per-turn isolated RNG are this experiment's adaptation, not a claim to reproduce
the original paper's exact settings. The upstream generation call inherits model
generation defaults; this implementation explicitly sets greedy decoding.

## User-owned smoke submission

From the DCGS root:

```bash
mkdir -p outputs/slurm
bash -n scripts/slurm/run_safedial_smoothllm_smoke.sbatch
sbatch --test-only scripts/slurm/run_safedial_smoothllm_smoke.sbatch
sbatch scripts/slurm/run_safedial_smoothllm_smoke.sbatch
```

One A40, researchshort, 64 GB RAM, 8 CPUs, 1 hour. Dialogue 1 has five turns:
40 candidate generations. Smoke wall time was 6m03s, with 324.706s summed candidate generation time
for five turns; peak allocated GPU memory was 13.91 GiB.
This is a separate allocation with no CAT dependency; simultaneous scheduling
depends on GPU availability and account limits. The agent must not submit jobs.

Output: `outputs/safedial_baseline/smoothllm_zephyr_smoke/`.
Logs: `outputs/slurm/safedial-smoothllm-smoke-<job-id>.out` and `.err`.
The batch performs generation, offline output/audit validation and judge dry-run
only. There are no paid API calls. Send the actual job ID for monitoring.

## Historical runner audit, resume and acceptance

`run_config.json` locks dataset, model, selection, decoder, detector strings,
randomness rules and local implementation hashes. Changes require a new output
directory. `answers.jsonl` has the selected answer in native format; `turns.jsonl`
also retains each perturbed current-user span, copy seed, full candidate response,
heuristic vote, selected index and per-copy token/runtime/truncation fields.
Top-level prompt/completion costs sum all copies. Reconstruct candidate inputs
by replacing only the final content in the saved original `prompt_history`.

Failures/empty candidates are errors, never safe votes. Partial candidate audits
are saved on handled turn failures. Interrupted, uncommitted dialogues replay
deterministically; completed dialogues are skipped. The identical batch retries
recorded error dialogues and compacts superseded records. A file lock prevents
concurrent writers from these batch invocations. Runtime stats cover the latest
invocation; retain Slurm logs/accounting for total cost across retries.

Accept the smoke only after Slurm COMPLETED/0:0, one dialogue/five valid turns,
40 valid audited candidates, successful dry-run, and runtime/peak GPU memory
inspection. Inspect any input truncation before scaling. The original staged plan called for smoke judging, 12-dialogue validation and
a 120-dialogue pilot. The user requested full generation on 2026-09-10; the
full launcher follows that direction. Those intermediate checks and safety
judging remain unrun; generation success does not establish safety quality.
Native safety scoring does not establish benign helpfulness or benign GCR.

Offline checks (no CUDA or API):

```bash
.venv/bin/python -B scripts/run_safedial_smoothllm.py --ids 1 --validate-only
.venv/bin/python -B -m unittest discover -s tests -q
```

Load `Python/3.11.11-GCCcore-13.3.0` first on this cluster. No new packages needed.

## Full generation (2026-09-10)

From the DCGS root, submit manually:

```bash
sbatch scripts/slurm/run_safedial_smoothllm_full.sbatch
```

All 2,037 dialogues / 10,029 turns / 80,232 candidates, retaining smoke inference
settings and pinned implementation. Separate output:
`outputs/safedial_baseline/smoothllm_zephyr_full`. One A40, 64GB RAM, eight CPUs,
48 hours in researchlong. No paid judging; final validator and judge dry-run
run only after generation returns successfully.

Extrapolating the five-turn smoke gives about 181 GPU-hours (7.5 days) of
generation. This is a rough planning estimate, not a validated full-run ETA;
response lengths and prompt lengths vary. The partition maximum is five days,
so the launcher uses resumable 48-hour allocations (roughly four at smoke speed,
possibly more). After a timeout, submit the same command again to resume; do not
submit overlapping copies. Completed dialogues are saved immediately and skipped.
An interrupted dialogue is regenerated. No automatic resubmission is configured.
Keep Slurm accounting/logs for total runtime across allocations; runtime_stats
only records an invocation that reaches the end.
