# CAT on SafeDialBench

> GPU request update (2026-09-10): the full launcher now requests one L40
> (48 GB) instead of A40, with the same 48-hour limit and inference settings.
> A40 smoke measurements below remain historical; full L40 timing is unmeasured.
> Editing the file does not update an already queued job. User must run
> `scontrol update JobId=245229 Gres=gpu:l40:1`
> while it is pending, then verify the request. No agent scheduler mutation.

Status: GPU smoke job 245201 PASSED on album (A40), exit 0:0, 1m46s.
One dialogue/five turns validated, zero input truncations; adapter logit delta
9.65625. Peak allocated 14.43 GiB / reserved 14.92 GiB. User-completed smoke
judging also passed: 5/5 calls, zero errors, complete=true; scores 8/7/8.
These are one-dialogue smoke scores only. The user explicitly approved moving
directly to the full CAT run, skipping the planned 12-dialogue check and pilot.
This changes CAT's execution staging, not its model, prompts or decoding.

## Full CAT run

From `DCGS`, the user can submit:

```bash
sbatch scripts/slurm/run_safedial_cat_full.sbatch
```

This generates all 2,037 English dialogues / 10,029 turns with the exact smoke
model/adapter pins and decoding settings, in a separate directory:
`outputs/safedial_baseline/cat_zephyr_full`. Smoke records are not copied or
overwritten. The full job repeats adapter-effect verification, records GPU
memory, validates final coverage/history/output integrity, and checks judge
input format without making paid calls.

Resources: one A40, 64GB RAM, 8 CPUs, 48-hour limit on `researchlong` (whose
current maximum is five days). A linear estimate from the five smoke turns is
about 34 GPU-hours; it is a weak estimate, not a promise. The 48-hour allocation
provides margin, but longer histories/output lengths could still require resume.

Rerun the same submission command after a failed/timed-out job has stopped to
resume saved dialogues and regenerate recorded error dialogues. A partially
generated dialogue may be repeated; complete successful dialogue records are
kept. A file lock prevents concurrent writers using this batch. Do not edit
inputs, config or weights between resumptions, and do not run the Python writer
directly while the batch is active. Runtime statistics describe the latest
invocation, so retain Slurm accounting/logs for total resumed-job cost.

Completion requires both Slurm success and a passing `validation.json` covering
all 2,037 dialogues / 10,029 turns. Review any reported input truncation before
claiming full parity. Paid full judging is a separate next step (10,029 successful
turn calls plus retries); a generation job alone produces no final safety score.

## Pinned artifacts

- Base: `HuggingFaceH4/zephyr-7b-beta`, revision
  `892b3d7a7b1cf10c7a701c60881cd93df615734c` (existing local weights).
- Adapter: [ContinuousAT/Zephyr-CAT](https://huggingface.co/ContinuousAT/Zephyr-CAT),
  revision `550ea10d3d0f867f62e205d928029573e0575e1b`.
- Adapter SHA-256:
  `48119f54d8a2db7e6613c28a3525d83d40fdb8c8805b66dfd8086e04ca9d51da`.
- Adapter size: 671,149,168 bytes; 448 finite tensors and 224 nonzero LoRA B
  matrices verified on CPU. Matches the release's LFS checksum.
- Lock: `configs/safedial/cat.lock.json`; staged files live in the project's
  `.cache/huggingface/hub`. Lock records config/weights/README hashes.

The adapter declares Zephyr as its base but does not pin a base revision; we
explicitly use our existing snapshot. Its model card does not declare a license;
retain its README and check upstream terms before redistribution. No retraining,
base-weight overwrite, model merging, or DCGS critic is involved.

## User-owned smoke submission

From the repository root:

```bash
bash -n scripts/slurm/run_safedial_cat_smoke.sbatch
sbatch --test-only scripts/slurm/run_safedial_cat_smoke.sbatch
sbatch scripts/slurm/run_safedial_cat_smoke.sbatch
```

Only the last command submits a real job. The test-only output may display a
prospective job number; it is not evidence of an actual submission.

The job requests one A40, 64 GB CPU RAM, eight CPUs and a one-hour ceiling.
It runs offline against dialogue 1 (five turns), writes to
`outputs/safedial_baseline/cat_zephyr_smoke`, and keeps the existing plain
Zephyr and DCGS outputs untouched.

Generation preserves native gold assistant history, uses the base Zephyr
tokenizer, applies the unmerged frozen CAT adapter, and uses seed 0, temperature
0.7, top-p 1.0, and 1,024 maximum new tokens per answer. The manifest includes
both base and adapter provenance. Plain-model manifests remain compatible with
existing runs.

Before generating, a fixed benign fixture checks that enabling versus disabling
the adapter produces different finite logits. This is a loading sanity check,
not an extra experimental arm or a safety/helpfulness result. Using unmerged
adapters avoids a merged/unmerged equivalence requirement.

## Pass criteria

- Slurm COMPLETED, exit `0:0`.
- `adapter_validation.json`: `passed: true`, active adapter, nonzero logit delta.
- `validation.json`: `passed: true`, one dialogue, five turns; zero unexpected
  truncations. The validator checks exact gold history, IDs, model labels,
  answer/turn agreement and no empty/error responses.
- `runtime_stats.json`: wall time and positive peak GPU allocated/reserved
  memory, including model loading and adapter verification. These are PyTorch
  allocator peaks, not total device use or Slurm CPU MaxRSS.
- Judge dry-run accepts the output schema for five turn judgments.

The batch does **not** make paid judge calls. Once generation passes, judge
the smoke separately when requested; do not infer safety from loading success.
The user has chosen to skip the intermediate check/pilot for CAT; see full-run
instructions above. Other methods retain their existing acceptance gates.

## Offline maintenance

```bash
LD_LIBRARY_PATH=/opt/apps/software/Python/3.11.11-GCCcore-13.3.0/lib \
  .venv/bin/python -B scripts/prepare_safedial_cat.py --offline
LD_LIBRARY_PATH=/opt/apps/software/Python/3.11.11-GCCcore-13.3.0/lib \
  .venv/bin/python -B -m unittest discover -s tests -v
```

To stage on a fresh machine, the preparation script without `--offline`
downloads only the pinned adapter, requires the pinned base already cached,
verifies files/tensors, and refuses to overwrite a differing artifact lock.
