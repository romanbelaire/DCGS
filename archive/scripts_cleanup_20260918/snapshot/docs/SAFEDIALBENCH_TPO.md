# Native SafeDialBench TPO

Current code uses parser **v6**: the [full-run continuation guide](SAFEDIALBENCH_TPO_FULL.md)
documents duplicate-block/exact-trailing-instruction recovery and prepared
`tpo_zephyr_full_v6/`.
The v3 smoke below is historical, validated with its original sources. Current
source hashes intentionally differ; preserve its completed output and manifest.

**The real GPU smoke passes.** Job 250500 completed 0:0 on A40; all five
turns / 75 candidates pass full audit and the judge dry-run. Final output is
`tpo_zephyr_smoke_v3/`. Three formatting warnings are recorded; none of those
candidates was selected. Paid safety judging has not run. The user-requested [full launcher](SAFEDIALBENCH_TPO_FULL.md)
is now prepared, including the explicitly documented native 8K reward context
needed by longer benchmark histories.
This is the planned Zephyr adaptation of Test-Time Preference Optimization,
not a reproduction of the original paper's 70B/22B model results.

## Method and artifacts

- Frozen actor: `HuggingFaceH4/zephyr-7b-beta`, revision `892b3d7a7b1cf10c7a701c60881cd93df615734c`.
- Frozen reward model: `sfairXC/FsfairX-LLaMA3-RM-v0.1`, revision `94fad49f1b3227aa8b566f415a335adb68ec544c`.
- Both are staged under `.cache/huggingface/hub`; paths and complete file SHA-256s are in [tpo.lock.json](../configs/safedial/tpo.lock.json). Reward shards match the official LFS digests; the nonzero finite classification head has shape `[1, 4096]`.
- Upstream algorithm reference: [official TPO](https://github.com/Simplified-Reasoning/TPO/tree/395c3d763a4c3df0ae72a4352b0db16fe5ee18e9), including `tpo_utils.py`, `reward_model.py`, and vendored TextGrad. Unmodified TextGrad prompt modules, source hashes, and MIT license are in `scripts/tpo_vendor/`.

The runner samples five initial responses, scores them with the reward model,
then performs two rounds of textual loss, textual gradient and five response
updates. Best and worst responses come from the entire scored cache; the final
answer is the highest-reward candidate. No model parameters are updated.
The SafeDialBench judge is not involved in candidate selection.

| Setting | Native TPO value |
|---|---|
| Width / refinement depth | N=5 / D=2 |
| Candidate decoding | Temperature 0.7, top-p 0.95, top-k disabled |
| Loss/gradient decoding | Temperature 0.7, top-p 0.99 |
| Candidate generation cap | 1,024 new actor tokens, including optimizer markup |
| Loss/gradient cap | 2,048 new actor tokens per call |
| Execution | HF Transformers, BF16, SDPA, serial batch size 1; both models on one GPU |
| Seed | 0, with deterministic dialogue/turn/stage/round/sample derivation |
| Output | One selected answer per benchmark turn |

Important adaptations are explicit in the manifest: native multi-turn gold
history; Zephyr acting as response and feedback model; local serial HF execution;
bounded feedback length (upstream server allows longer feedback); independently
seeded calls; and indexed duplicates with first-index tie handling. Initial actor
and reward calls receive native role messages. Feedback includes a role-tagged
serialization of the same visible history. Future/reference answers and task
labels are never passed to a model.

Optimizer updates require exactly one opening `<IMPROVED_VARIABLE>` tag
before the closing tag (or output end), and nonempty extracted text. A missing closing tag is accepted with a warning;
extraction continues to the end of the output. An opening-tag reference after a completed span produces a trailing-text
warning. Nested openings, multiple/reversed closing tags, missing openings
and empty answers still fail. Candidate counts remain unchanged. The scored/extracted text is also the final evaluated text;
there is no separate hidden clipping of selected responses. Neither actor nor
reward inputs are truncated. Dynamic context overflow fails before execution.
The upstream first/last-ten-word display inside feedback prompts is retained;
the full chosen response is also present in the feedback conversation context.

## Smoke and manual execution

The smoke uses dataset dialogue **1: five turns**, with **75 scored candidates,
20 loss/gradient generations, and 75 reward forwards**. Total actor generations
are 95. This tests method execution and provenance, not a benchmark safety score.

The completed launcher targets v3 and reused all 163 saved calls before
executing its final three generations and four rewards. Repeating it resumes
the completed output without new generation. The manual command used was:

```bash
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
sbatch scripts/slurm/run_safedial_tpo_smoke.sbatch
```

The batch requests **one A40 (48 GB), 64 GB CPU RAM, 8 CPUs and two hours** in
`researchshort`. Both models use that one GPU, sequentially at batch size one.
Across the three generation allocations, peak allocated/reserved memory was
28.51/29.09 GiB on this GPU. Full smoke validation passed; these measurements
cover this one dialogue, not every full-benchmark prompt. No Slurm submission
was performed by the agent.

The launcher runs generation, full offline audit validation and judge
`--dry-run`. It makes no paid API calls and does not run safety judging.
Return the actual job ID for completion/log/output checks.

Expected paths:

```text
outputs/slurm/safedial-tpo-smoke-<JOBID>.out
outputs/slurm/safedial-tpo-smoke-<JOBID>.err
outputs/safedial_baseline/tpo_zephyr_smoke_v3/
  continuation.json     # exact import policy, source hashes and reused calls
  provenance/249814/    # original V1 manifest, journals, runtime and source files
  provenance/249899/    # original V2 files and exact import/preparation script
  run_config.json       # model hashes, algorithm, source hashes, decoding
  preflight.json        # tokenized gold-history checks
  model_loading.json    # no missing/random weights; zero trainable parameters
  events.jsonl          # durable calls, exact inputs, raw outputs, rewards, costs
  turns.jsonl           # turn results, candidates, rounds and audit traces
  answers.jsonl         # only complete error-free dialogues
  invocations.jsonl     # actual calls/resources for each invocation
  runtime_stats.json    # latest invocation that loaded models
  validation.json       # created only after the complete output passes
```

## First GPU attempt and startup repair

Job **249773** failed on avenue after **57 seconds**, exit **2:0**, with
`Invalid device argument `. Artifact hashes and tokenizer preflight passed;
no model loaded and no event, turn or answer was generated. The runner reset
peak-memory counters with explicit `cuda:0` before initializing the allocator.
The installed PyTorch reproduced the exact error while CUDA remained
uninitialized. The runner now calls CUDA init, selects the logical device,
then resets the counters. Startup errors also save their stage and runtime
records without querying an uninitialized allocator.

The failed attempt is preserved at
`outputs/safedial_baseline/tpo_zephyr_smoke_failed_249773/`, including the old
manifest, preflight, failure summary and hash-verified original source files.
The subsequent attempt 249814 verified CUDA initialization and model loading.
Model/decoding settings and batch resources were unchanged.

## Corrected GPU attempt 249814

The corrected runner loaded both models on A40, with complete weight loading
and zero trainable parameters. Job 249814 failed after **13m52s**, exit 2:0.
**Two of five turns passed** the full selection/history/seed/journal audit.
Turn index 2, update round 1, slot 2 opened `<IMPROVED_VARIABLE>`, then copied
feedback-context markup and reached 1,024 tokens without the closing tag.
The strict parser rejected it. Two later turns were not attempted; no complete
dialogue was exported. Increasing the cap alone is not a verified fix.

There were **55 actor calls and 42 reward calls**, zero input truncations and
one cap hit. Peak allocated GPU memory was **28.26 GiB**, reserved **28.72 GiB**;
runner elapsed time was 803.66s excluding artifact hashes/preflight. This
establishes model loading and memory fit on the observed prompts, not a passed
smoke or full-run throughput estimate. Final validation and judging did not run.

Partial results and the exact manifest remain at `tpo_zephyr_smoke/`, with
`smoke_failure_review.json` and hash-verified source snapshots under
`provenance/249814/`. The original directory remains unchanged. The v2
continuation below explicitly imports its results under the approved parser policy.

## Format policy v2 and verified continuation

User approved relaxing only the missing-closing-tag check. The pinned upstream
optimizer uses split(start)[1].split(end)[0], which also accepts the remainder
when the end tag is absent. Our v2 parser records
`missing_IMPROVED_VARIABLE_closing_tag; extracted_to_end_of_output` in the
generation event's `format_warnings`. The full validator re-derives that warning
from the raw response and reports `optimizer_missing_closing_tag_warnings`.

The entire extracted body is preserved for reward scoring and, if selected,
for the final answer. No closing tag is fabricated and no copied context is
removed. This can affect response quality; it remains visible in the output.
The 1024-token cap, N=5/D=2, prompts, seeds, reward model and candidate budget
are unchanged. No error-feedback retries were added. Other SafeDial runners
and launchers were verified unchanged by hashes.

The offline import into `tpo_zephyr_smoke_v2/` checks the exact old manifest and
journal hashes, confirms all other settings and model pins match, and replays
the successful turn audits. It retains two completed turns and all 97 model
calls. Only event (dialogue 1, turn index 2, event 28) changes classification
from a parser error to the derived formatting warning. Every saved request and
raw model result is unchanged. The unfinished turn is represented by its event
journal and resumes directly at the pending reward call. The old failed turn
record is preserved in provenance.

`continuation.json` records this import. Original invocation measurements are
retained with `source_job_id=249814`; future `runtime_stats.json` measures only
the continuation. Sum `invocations.jsonl` for actual calls across allocations;
the completed turn audits include all calls used for generation. Source and
output snapshots plus the one-time preparation script are under
`provenance/249814/`. No inference was performed during preparation.

## Continuation 249899 and trailing-tag correction v3

Job 249899 failed after **9m10s** with **four of five turns passing**. The
missing-close candidate was accepted and scored (-2.828125), but not selected
(selected reward 2.203125). The final turn produced a complete tagged answer,
then echoed an opening tag in a trailing formatting instruction. At 540 tokens,
this was not a token-cap failure. V2's global opening-tag count rejected it.

V3 checks opening-tag uniqueness only before the closing tag. The answer inside
the completed span is unchanged; trailing instruction text is excluded, as
ordinary trailing text already was. The event records
`opening_IMPROVED_VARIABLE_tag_after_closed_span; ignored_trailing_text` and
the validator reports `optimizer_trailing_opening_tag_warnings`. Missing-close
warnings remain supported; multiple completed spans and nested openings remain
errors. The model-call budget, prompts, seeds, caps and rewards are unchanged.

V3 imports all 163 saved requests/results, preserving four completed turns and
reclassifying only event (1,4,26). Exact source hashes are in continuation.json;
V1/V2 source/output copies and import scripts remain in provenance. Existing
invocation costs retain source job IDs 249814 and 249899. V3 resumes at the
pending score, with only three generations and four reward forwards remaining.
Original V1/V2 output directories are unchanged.

249899 measured peak allocated/reserved GPU memory of 28.51/29.09 GiB and
521.48s runner time, with 37 new generations and 29 reward calls. Across both
generation allocations: 92 generations and 71 rewards. These are partial-run
measurements; the full smoke and judge dry-run remain unpassed.

## Successful completion 250500

Job **250500 COMPLETED 0:0 in 1m39s** on avenue. Final generation validation
with `--require-gpu` and judge `--dry-run` passed. Output contains one complete
dialogue, five successful turns, 75 scored candidates and 20 feedback generations.
There are 170 unique audited calls: **95 actor generations and 75 reward calls**
across all three generation allocations. No execution errors or input truncations.

Warnings: one missing closing tag and two opening-tag references after completed
spans; one generation hit its token cap. All three warned candidates were scored,
and none became a selected answer. Candidate budgets and actual invocation call
counts agree exactly, confirming no generation retries or duplicate calls.
Original V1/V2 file hashes still match their continuation records.

Peak allocated/reserved memory across allocations is **28.51/29.09 GiB**. Total
wall time for 249814 + 249899 + 250500 was **24m41s**, including repeated loading
and preflight. Summed runner time was 1394.44s, including loading but excluding
artifact hashing/preflight. The failed CUDA-startup allocation 249773 is excluded
from those generation totals. The last job's 1m39s covers only its continuation.
These are smoke measurements, not a full-benchmark throughput estimate.

`smoke_review.json` records the complete audit, cost scope and each warned
candidate's selection status. `validation.json` is the full validator result.
Paid judging has not run; the dry-run validated five required judgment requests.
This is a passed technical smoke, not a safety-evaluation result.

## Validation and recovery

Successful events are fsynced immediately and reused after interruption.
Successful turns are also retained. An incomplete last JSONL append is recovered
under the output lock, with its exact original tail saved in `recovery/`.
Malformed interior records are rejected. Run/model/source changes refuse resume.
The runner owns its own nonblocking file lock, including direct CLI invocations.

An ordinary interruption resumes with the same batch. A recorded model/format
error stops the run; inspect its event audit first. Explicit `--retry-errors`
allows retrying the failed call while reusing prior successful steps. The smoke
batch deliberately does not enable automatic error retries. Identical seeds can
reproduce a format failure, so repeated retries are not a diagnosis.

After actual completion, validate with:

```bash
module load Python/3.11.11-GCCcore-13.3.0
.venv/bin/python -B scripts/validate_safedial_tpo.py \
  --output-dir outputs/safedial_baseline/tpo_zephyr_smoke_v3 --require-gpu
```

The smoke loading, reward/selection/history, format-warning, token-cap and
resource checks have passed. Final safety metrics still require separate judging.
The [full TPO launcher](SAFEDIALBENCH_TPO_FULL.md) is now prepared with separate
output, native 8K reward context and a GPU startup capacity probe.

## Completed offline checks

- All **61 repository tests PASS**, including 19 TPO tests and a real tiny
  CPU classifier-forward check. New coverage includes cumulative selection,
  ties/duplicates, gold history, malformed outputs, nonfinite rewards, context
  overflow, audited cost totals, tampering, mid-turn event resume, failed-call
  retry, interrupted append recovery, exclusive writer locking, cold CUDA
  initialization, saved startup failures, GPU-free validate-only mode, missing
  closing tags with/without cap hits, unchanged reward/final text, warning
  tampering, full-validator warning counts and no extra generation retries.
- Real actor/reward artifact hashes and tokenizers PASS. Smoke gold-prefix
  maxima: actor 1,753 tokens, reward 1,535 tokens. Effective context limits:
  actor 32,768; reward 4,096 (minimum of model config and tokenizer metadata).
  Candidate/feedback-dependent input lengths are checked at each runtime call.
- V3 continuation fixture PASS: reuses the real 163-event prefix, injects only
  the final seven calls, and completes all five turns / 75 candidates, with
  one missing-closing-tag warning and one trailing-opening-tag warning.
  Full audit PASS. Stored at `/tmp/tpo-v3-resume-fixture-6aup3jsq` and clearly
  marked `OFFLINE_FIXTURE_NOT_BENCHMARK.txt`; this is not a GPU result.
- `bash -n scripts/slurm/run_safedial_tpo_smoke.sbatch` PASS.
- No new dependencies, training, full-size model loading, GPU inference,
  paid judging, or job mutations were performed during preparation.

Artifact restaging/checking is available through `scripts/prepare_safedial_tpo.py`.
Use `--offline` to recheck already-staged artifacts; GPU jobs remain offline.
