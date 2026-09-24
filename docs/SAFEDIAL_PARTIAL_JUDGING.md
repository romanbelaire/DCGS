# Partial SafeDial judging of DCGS runs

## 384-token snapshot, 22 September 2026

Prepared and validated `outputs/safedial_partial_judging/dcgs_384tokens_20260922_v1`
from the current belief-only 384-token outputs. VDCGS generation can continue;
RDCGS generation was preempted. The judge reads immutable copies, not live exports.

| Method | Complete dialogues | Turn judgments |
|---|---:|---:|
| VDCGS-384 | 923 | 4,462 |
| RDCGS-384 | 497 | 2,403 |

The batch makes 6,865 GPT-4o-mini requests before retries, with four concurrent
requests per method and the same native rubric/settings as the earlier evaluation.
Cached Zephyr scores cover every selected ID. The main three-way comparison uses
496 shared dialogue IDs; each method's additional completed dialogues are also
judged and reported separately. This is a preliminary completed-prefix sample,
not a full-benchmark result or a direct causal comparison with the 96-token runs.

Submit from the DCGS root (CPU only; no GPU request):

```bash
sbatch scripts/slurm/run_safedial_dcgs_partial_judge.sbatch \
  outputs/safedial_partial_judging/dcgs_384tokens_20260922_v1
```

The same command resumes successful turn judgments without repeating them.
Native Identification/Handling/Consistency scores use per-dialogue minima; this
batch does not run LlamaGuard or compute combined safety pass rates.
Results are written to `comparison.json` in the snapshot directory.

Preparation checked answer/journal equality, model identity, gold histories,
384-token configuration, cached-baseline provenance and all snapshot hashes.
VDCGS terminal failures at 178/turn4 and 903/turn4 remain coverage exclusions;
RDCGS has no recorded generation failures. Incomplete dialogues are excluded.
The preparation program is retained as `preparation.py` inside the snapshot.
Both native judge dry runs, four partial-judge regression tests and batch shell
syntax passed. No paid API requests or Slurm submissions during preparation.

## 96-token snapshot, 20 September 2026

User requested judging processed outputs before deciding whether to change the
belief budget. Generation jobs remain user-owned and may keep running; judging
reads a frozen snapshot, never their growing answer files.

Prepared snapshot: `outputs/safedial_partial_judging/dcgs_96tokens_20260920`.
VDCGS457fully completed dialogues/2218turns; RDCGS389/1886. Total4104 new paid
GPT-4o-mini requests before retries, CPU-only. Existing Zephyr baseline scores
cover every snapshot ID and are reused without API calls. Both frozen answer
files were checked against the dataset and corresponding successful turn journals.

The same official per-turn rubrics and per-dialogue minimum aggregation are used
as for the baseline: judgegpt-4o-mini, temperature0.7, max_tokens2048, seedNone,
choice0. Existing dataset/prompts/answer provenance was verified. Frozen files,
judge scripts and dataset hashes are checked before API calls.

Submit manually from the DCGS root:

```bash
sbatch scripts/slurm/run_safedial_dcgs_partial_judge.sbatch \
  outputs/safedial_partial_judging/dcgs_96tokens_20260920
```

This is one CPU job, running VDCGS then RDCGS with four concurrent judge requests
within each method. Reads existing .env/OPENAI_API_KEY as the usual judge does;
no credential is copied into the snapshot. Slurm logs:
`outputs/slurm/safedial-dcgs-partial-judge-JOBID.{out,err}`.

Each method gets its own `judgments_gpt-4o-mini/` directory beneath the snapshot.
The judge resumes already successful judgments if the same snapshot is rerun.
`comparison.json` contains matched three-way scores, matched task breakdowns,
each method versus Zephyr on identical IDs, and a descriptive grouping by whether
the selected belief lacks an Instruction: field. It is finalized after both judge
subprocesses finish. A partial report can be regenerated without API requests:

```bash
module load Python/3.11.11-GCCcore-13.3.0
.venv/bin/python -B scripts/judge_safedial_partial.py report \
  --snapshot outputs/safedial_partial_judging/dcgs_96tokens_20260920
```

Require `complete_for_frozen_snapshot: true` for a final snapshot result.
`complete_for_full_benchmark` remains false. The main three-way comparison uses
only IDs shared by VDCGS, RDCGS and baseline; pending judgments are reported.

Two VDCGS generation failures (155/index0,408/index2) are recorded as coverage
exclusions, not silently treated as successful or assigned arbitrary safety
scores. Incomplete dialogues and later appended generation records are outside
this frozen snapshot. The completed-prefix sample is not random and may have
survivorship/category bias. A selected belief missing Instruction: is a structural
flag, not a complete semantic classification. Group associations and differences
from Zephyr do not establish a causal effect of96tokens; that requires a matched
larger-budget experiment. New judge calls also occur at a different time than the
cached baseline; its source-model alias/backend may have changed.

Validation: four regression tests pass (matched IDs, pending score coverage,
append-only snapshot boundary, changed-input rejection), shell syntax passes,
and both official judge dry runs pass. No paid judge calls or Slurm submissions
were made during preparation. Existing generation code/configurations untouched.
