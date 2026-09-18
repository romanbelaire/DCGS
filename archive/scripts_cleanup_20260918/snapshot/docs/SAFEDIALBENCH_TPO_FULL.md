# Full SafeDialBench TPO run

Updated 2026-09-17: parser v7 uses the first improvement, whether or not it
has a closing tag. A preserved continuation of job 254463 is prepared; GPU
continuation is pending user submission. GPU smoke 250500 previously passed.
Generation covers all **2,037 dialogues / 10,029 turns**, seed 0, N=5/D=2,
1,024-token candidate cap and 2,048-token feedback cap. Frozen actor/reward
weights and prompts match the [completed smoke](SAFEDIALBENCH_TPO.md).
Smoke safety judging and the intermediate pilot were not run before this
user-requested full-run preparation. No paid evaluation is included.

From DCGS:

```bash
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
sbatch scripts/slurm/run_safedial_tpo_full.sbatch
```

The batch requests **one A40, 64 GB RAM, eight CPUs and 48 hours** in
`researchlong`. The launcher uses the established 48-hour allocation size. Multiple allocations may be
needed. After an interruption or time limit, rerunning the same command resumes
saved events and turns. Do not submit overlapping copies. The runner owns an
exclusive output lock. Recorded execution/format errors stop for inspection;
the launcher does not blindly retry them. All Slurm execution is user-owned.

Output: `outputs/safedial_baseline/tpo_zephyr_full_v7/`. It imports 130
successful turns, 25 complete dialogues and all 4,439 saved calls from v6
job 254463 (including earlier full-run imports). All original full-run
directories remain unchanged. No smoke outputs are imported.
Stdout/stderr: `outputs/slurm/safedial-tpo-full-<JOBID>.{out,err}`.

## Full-dataset context check

The original smoke entrypoint conservatively uses the minimum of model and
tokenizer limits. Full preflight found **257 gold-history prefixes exceeding
4,095 reward tokens**, first at dialogue 53 / turn index 5 (4,252 tokens).
The maximum prefix is **7,197 tokens**, dialogue 987 / turn index 5.

The pinned reward [model configuration](https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1/blob/94fad49f1b3227aa8b566f415a335adb68ec544c/config.json)
specifies `max_position_embeddings=8192`, while its tokenizer metadata says
4096. The separate full entrypoint uses the model's native **8,192-position**
limit, without changing weights or RoPE configuration. Its manifest explicitly
records this context policy and the full entrypoint's source hash. The original
smoke entrypoint, generation runtime and completed output remain unchanged.

At startup, after both frozen models load, the full runner executes one
synthetic **8,192-token reward forward** and requires a finite scalar result.
This tests capacity with both models resident. Failure stops before any missing
benchmark model call. The result is saved in `context_probe.json`; it is never
used to rank candidates. Its call count and latency are recorded separately
from optimization calls. This full-size GPU probe runs in the user's job;
it has not been executed on the login node.

All full-dataset gold prefixes are checked offline. Every actual actor and
reward input is also checked before execution. Candidate text uses a different
tokenizer from the reward model, so a passing prefix check does not guarantee
every later combined input fits. Inputs above their supported limit stop with
an explicit error; the runner never truncates history or candidate text.
Architectural capacity does not establish reward quality on longer histories;
the full evaluation will measure the resulting method's behavior.

## Execution and accounting

The full runner reuses the validated TPO algorithm, resume journal and HF
backend. Per turn it performs 19 actor generations (15 candidates and four
feedback generations) and 15 reward forwards. For the complete dataset this
is **190,551 actor generations and 150,435 candidate reward forwards**, plus
one separately recorded context probe per allocation that loads models.

Formatting policy v7 takes text after the first `<IMPROVED_VARIABLE>` up to
the next opening tag, closing tag, or end of output, then strips surrounding
whitespace. Later answers are ignored even if they differ. A missing opening
tag or empty first span remains an error; the parser never skips an empty
first span to choose a later answer. Tags are literal boundaries, including
any quoted tags. With no later delimiter, all remaining text is the candidate,
including any trailing commentary or text cut off at the generation cap.

The raw output stays in the audit and the extracted candidate receives normal
reward scoring and selection. Historical v6 warning labels are preserved for
previously accepted responses so successful saved audits replay unchanged.
Newly accepted malformed outputs record
`extra_or_misordered_IMPROVED_VARIABLE_tags; extracted_first_improvement_only`;
the validator counts these as `optimizer_first_improvement_warnings`.
Prompts, weights, decoding, seeds and budgets are unchanged.

`runtime_stats.json` covers the current generation invocation, including model
loading and its capacity probe. `invocations.jsonl` preserves allocation-level
costs. Successful events are fsynced immediately, and only complete error-free
dialogues are exported to `answers.jsonl`. A time limit can leave valid partial
work without a final validation file; resumption reuses that saved work.

After generation completes, the batch runs:

```bash
.venv/bin/python -B scripts/validate_safedial_tpo_full.py \
  --output-dir outputs/safedial_baseline/tpo_zephyr_full_v7 --require-gpu
.venv/bin/python -B scripts/judge_safedial.py \
  --answers outputs/safedial_baseline/tpo_zephyr_full_v7/answers.jsonl --dry-run
```

The full validator checks source hashes, context policy, capacity-probe result,
frozen-model loading, complete coverage and the same candidate/selection/cost
audit used by the smoke. Paid judging is a separate subsequent action.

To request an A100 as in job 252269, use:

```bash
sbatch --gres=gpu:a100:1 scripts/slurm/run_safedial_tpo_full.sbatch
```

## V7: first-improvement extraction and continuation

Job 254463 stopped at dialogue 26/index 3, update round 0/slot 3, event 18.
Its capped 1,024-token output has two opening tags and no closing tags, with
different answer text after each opening. V7 extracts only the first answer.

`scripts/prepare_safedial_tpo_v7.py` verifies the original source/output hashes
and imports into a fresh directory. Only event `(26, 3, 18)` is reclassified;
all 130 successful turns and 4,439 calls (2,481 actor / 1,958 reward) are reused.
`provenance/254463/` retains exact v6 outputs/sources and nested provenance.
The next new benchmark call scores the recovered answer; no regeneration or
`--retry-errors` is needed. `continuation.json` records the new extraction policy.
Original run directories retain their original manifests and parser versions.

## Historical v6:  equivalent duplicate blocks followed by the exact formatting instruction

Job 254257 completed the previously recovered turn (25/index 3), then failed
at dialogue 25/index 4, update round 0/slot 2, event 16. The saved 799-token
response contains two identical complete blocks, followed by:

```text
Send ONLY the improved variable between the <IMPROVED_VARIABLE> tags, and nothing else.
```

V6 permits that exact suffix, with only boundary whitespace ignored, after two
complete non-nested blocks that satisfy the existing v5 equivalence rule.
It preserves the first block's text for scoring and records both
`equivalent_duplicate_IMPROVED_VARIABLE_blocks; extracted_first_block` and
`opening_IMPROVED_VARIABLE_tag_after_closed_span; ignored_trailing_text`.
Raw generation remains unchanged in the audit. Additional suffix words, altered
case/internal whitespace, repeated instructions, conflicting bodies, nested tags,
third blocks or an unexplained extra opener still fail. All previously accepted
formats retain their prior extraction and warnings. No prompts, decoding,
stopping conditions, token caps, model weights, seeds or voting/selection change.

`scripts/prepare_safedial_tpo_v6.py` verifies the original source/output hashes
and imports into a fresh directory. Only event `(25, 4, 16)` is reclassified;
126 successful turns and 4,301 calls (2,404 actor / 1,897 reward) are reused.
`provenance/254257/` retains exact v5 outputs/sources and nested provenance.
The next new benchmark call scores the recovered answer; no regeneration or
`--retry-errors` is needed. `continuation.json` pins the import.

Offline replay injected only 17 missing fixture calls to complete the first
25 dialogues (127 turns / 1,905 candidates). Full source/history/selection/cost
validation passed, with two duplicate-block warnings and seven trailing-opening
warnings across this fixture, and byte-identical no-op resume. The benchmark
output itself still contains only the original 126 completed turns; injected
fixture outputs remain in `/tmp`. See v6 `offline_preparation_review.json`.

## Historical duplicate-block repair and v5 continuation

Job 252269 stopped at dialogue 25 / turn index 3, update round 0 / slot 2,
event 16. Its 1,024-token output contains two complete answer blocks and
instruction echoes; the second answer repeats the first with outer braces.
Both closing tags precede the cutoff. The previous parser rejected the second
closing tag. Increasing the token budget would not remove that ambiguity.

V5 accepts exactly two complete, non-nested blocks, with no extra optimizer
tags, only when the bodies match after stripping boundary whitespace and at
most one surrounding literal brace pair from each body for comparison. It does
not normalize internal whitespace, punctuation or case. The normalized body
must be nonempty. The first body's text (boundary whitespace stripped, braces
otherwise retained) is scored unchanged. Raw output stays in the audit and the
event receives `equivalent_duplicate_IMPROVED_VARIABLE_blocks; extracted_first_block`.
Conflicting bodies, nested/extra tags and three or more blocks still fail.
Existing single-block and terminal-opening policies remain unchanged. This is
a documented extraction adaptation; prompts, seeds, weights, sampling, token
caps and reward-based candidate selection are unchanged.

`scripts/prepare_safedial_tpo_v5.py` performs a pinned offline import and refuses
changed input hashes or an existing destination. V5 `continuation.json` records
4,267 reused calls (2,385 actor / 1,882 reward), 125 successful turns, and the
sole reclassified event `(25, 3, 16)`. `provenance/252269/` contains exact v4
output/source copies and its nested original-job provenance. The failed turn
is represented by reusable events until completed. Its next new call is reward
scoring of the recovered answer; no regeneration or `--retry-errors` is needed.
Runtime/old invocations remain traceable to their source runs. New errors still
stop for inspection. The v5 continuation was submitted as 254257 and failed on the combined
duplicate/trailing-instruction case now handled by v6.

Offline replay reused all 4,267 real calls and injected 51 fixture calls to
complete dialogues 1–25: full audit passed for 127 turns / 1,905 candidates,
one duplicate-block warning, and a no-op resume with byte-identical journals.
Fixture data lives only under `/tmp`; it is not benchmark output. See v5
`offline_preparation_review.json` for the recorded checks. The real prepared
output still has 125 completed turns / 24 dialogues and no new model calls.

## Historical terminal-opening failure and v4 continuation

Job 250576 failed at dialogue 2 / turn index 4, update round 0 / slot 1.
The 803-token response started and ended with `<IMPROVED_VARIABLE>` and had
no closing tag. It did not hit the token cap or truncate input. Both frozen
models loaded and the real 8192-token GPU probe passed before generation.

The historical v4 `continuation.json` pins the original file hashes and records the sole
reclassified event `(2, 4, 14)`. Original files, runtime measurements and
hash-verified source snapshots are under `provenance/250576/`. Preparation
uses `scripts/prepare_safedial_tpo_v4.py`; it refuses altered source data or an
existing destination. Its snapshot argument can use the archived
`provenance/250576` directory for reproducing the import into a fresh target.
The original invocation is retained with its source job/run IDs. The failed
turn stays incomplete until the remaining calls finish; the recovered raw
response is reused and its next call is reward scoring. No `--retry-errors`
is needed for this prepared continuation. New errors still stop for inspection.

Offline replay reused all 321 real calls and injected only the 19 missing calls
to finish dialogues 1 and 2. Full audit passed for 10 turns / 150 candidates,
including one terminal-opening warning, and no-op resume made no new calls.
This is a fixture in `/tmp/tpo-v4-offline-continuation-1waeqssc`, not benchmark
output. The real v4 continuation was submitted as 252269 and stopped at the
duplicate-block case described above. Full safety judging has not started.

## Preparation verification

- All **85 repository tests pass**, including native context-bound checks, a
  tiny real CPU classifier capacity probe, separate probe accounting and
  stopping before benchmark calls when the probe fails.
- Full-dataset artifact/tokenizer preflight **passes**: 2,037 dialogues,
  10,029 turns, 150,435 planned candidates. Maximum actor/reward gold prefixes
  are 8,022/7,197 tokens. [Saved preflight evidence](SAFEDIALBENCH_TPO_FULL_PREFLIGHT.json).
- Full batch `bash -n` passes.
- Completed v3 GPU smoke was validated with its original sources. Current
  v6 sources deliberately do not match old manifests; do not overwrite them.
- Active SmoothLLM source hashes match its saved manifest.
- No Slurm submission, full-size model loading, GPU inference or paid calls
  were performed during full-run preparation.

Offline preflight command:

```bash
.venv/bin/python -B scripts/run_safedial_tpo_full.py \
  --seed 0 --sample-size 5 --max-iters 2 \
  --max-new-tokens 1024 --feedback-tokens 2048 \
  --output-dir outputs/safedial_baseline/tpo_zephyr_full_v7 --validate-only
```
