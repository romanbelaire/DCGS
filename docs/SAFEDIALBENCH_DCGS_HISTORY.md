# SafeDial DCGS history compatibility

`scripts/run_safedial_dcgs_context.py --include-history` enables the original
low-level generator's `ll_action_belief_only=False` mode for VDCGS and RDCGS.
Omitting the flag selects belief-only mode in this separate experiment runner.
The existing v3 runner, configuration, sources and results stay intact.

The candidate-list and fallback prompts include the selected belief, every
supplied prior gold conversation turn, and the current user message. They never
include the current turn's reference answer or future turns. Previous generated
answers remain saved/scored outputs and do not become subsequent history.

Only the low-level generation context flag changes. Belief generation remains
96 output tokens for the five-item list; response generation remains 640 tokens
for the candidate list and 128 for its fallback. The high-level critic still
left-truncates its separate input to 1,500 tokens. The low-level critic already
receives history, belief and candidate response, and still rejects combined
inputs over 8,192 tokens without truncating them.

The actor uses the original tokenizer behavior and input-plus-output capacity
check. With the pinned Zephyr tokenizer there is no configured tokenizer cutoff;
this feature adds no trimming or 1,024-token input cap. An actor context overflow
still stops the run as an integrity error. The 32,768-token preflight check uses
synthetic beliefs/responses and is an estimate; runtime checks are authoritative.
Protecting the belief while trimming only oldest history would be another policy
change and is not part of this compatibility feature.

## Smoke commands

From the DCGS root, submit either or both:

```bash
sbatch scripts/slurm/run_safedial_dcgs_history_smoke.sbatch vdcgs
sbatch scripts/slurm/run_safedial_dcgs_history_smoke.sbatch rdcgs
```

Each requests one A100 for up to four hours and runs dialogue 1, including all
its turns. The batch performs tokenizer preflight, inference, saved-event replay,
GPU loading/memory validation, and judge dry-run. It makes no paid judge calls.
Outputs are `outputs/safedial_dcgs/zephyr_{vdcgs,rdcgs}_main_history_smoke_v1`.
`DCGS_ARTIFACT_LOCK` can select an already verified alternative artifact lock.
Inspect the completed smoke's integrity, runtime and memory before a full run.

For CPU-only preflight of another selection:

```bash
module load Python/3.11.11-GCCcore-13.3.0
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B \
  scripts/run_safedial_dcgs_context.py --method vdcgs --include-history \
  --ids 1 --device cpu --validate-only \
  --output-dir outputs/safedial_dcgs/history_preflight_cpu_v1
```

Use a different directory for GPU generation because device is recorded in the
manifest. The runner accepts the existing `--ids`, `--limit`, `--per-task`,
`--seed`, and `--on-turn-error` options.

## Identity, audit and continuation

Context experiments use protocol `safedial_main_wildjailbreak_ll_context_v1`,
distinct model IDs for each method/context mode, and saved effective settings
and source hashes. Turn seeds retain the original model-ID-based derivation;
the same CLI seed across distinct model IDs does not imply identical turn RNG
seeds. Do not describe these runs as an exactly matched per-turn seed experiment.

An existing output directory cannot switch context mode or accept original v3
results. Use this entry point to audit its outputs, with mode read from the
saved manifest:

```bash
.venv/bin/python -B scripts/run_safedial_dcgs_context.py \
  --output-dir outputs/safedial_dcgs/zephyr_vdcgs_main_history_smoke_v1 \
  --audit-only --require-gpu
```

Use `--allow-incomplete` to audit partial coverage without treating it as complete.
Rerunning the identical generation command resumes missing turns; completed
turns are replay-audited and a completed run performs no additional inference.
The existing `--recover-from SOURCE --recovery-reason REASON` preparation is
supported for inspected infrastructure failures, with the same settings and a
fresh target directory. It preserves source records and never retries terminal
model-output failures. Context-mode changes are new experiments, not recovery.

## Offline verification

```bash
module load Python/3.11.11-GCCcore-13.3.0
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B -m unittest discover -s tests -p 'test_safedial_dcgs*.py' -v
bash -n scripts/slurm/run_safedial_dcgs_history_smoke.sbatch
```

Tests exercise history in both candidate and fallback prompts, reference-answer
isolation, unchanged critic inputs/limits, wrong-mode replay rejection, saved-run
audit and no-op resume, and cross-mode output protection. CPU evidence establishes
compatibility, not answer quality or GPU runtime success.
