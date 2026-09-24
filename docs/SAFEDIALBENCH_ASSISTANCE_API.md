# Assistance judging with the native labeling model

The user requested `gpt-4o-mini`, the model recorded in the existing native
SafeDialBench judge configuration, with **eight concurrent API workers** and
the **same assistance prompt**. This evaluates assistance, not the native
Identification/Handling/Consistency rubric. The failed Zephyr pilot and its
source remain unchanged.

`scripts/judge_safedial_assistance_api.py` reuses the existing frozen-input,
goal-provenance, strict-label and saved-LlamaGuard-join implementations. It has
a separate API protocol and output directory; old local/API results cannot be
mixed in the same manifest.

## Unchanged prompt, explicit API framing

Every request contains exactly:

```python
messages = [{
    "role": "user",
    "content": render_prompt(goal, user_message, response) + "[ASSIST]",
}]
```

`render_prompt` is imported unchanged from the local assistance runner. There
are no additional system instructions, examples, JSON schema, chat-template
edits, text clipping, goal changes or substituted native scoring rubrics.
The last `[ASSIST]` is literal user-message text; the API generates a new
assistant message rather than continuing a supplied assistant token prefix.
This matches the repository API helper's way of packaging a raw prompt and
suffix, while retaining the **local** assistance rubric rather than its shorter
API-specific rubric.

Generation retains temperature 0 and the 24-token output limit. The normalized
runner accepts exactly YES or NO after Unicode/case normalization and removal
of punctuation, symbols, whitespace and known ASSIST/numbered/fence wrappers.
Explanations, negations, conflicting labels, refusals, filtered responses and
token-limit finishes remain errors.
The inherited adversarial-goal assumption also remains: extracted goals include
benign/protective requests, so a YES is not independently verified harmfulness.

## Pilot result and launcher

User-submitted job **302438** ran on24September15:07:01–15:07:07 SGT,
using eight CPU workers onhammer. It returned14 directly parsed labels and
one clear NO in the exact numbered format from the prompt. The original
parser rejected numbering, causing Slurm exit2. The API-specific parser now
accepts that precise format while rejecting extra prose or conflicting labels.
The unchanged saved responses were revalidated offline into a separate v2
directory: **15/15 valid, 3YES and12NO; no additional API calls**. Original v1
artifacts and the failed job status remain intact. The v2 protocol versions
only the parser change; prompt/model/request settings are unchanged.
See [audited reparse](verification/assistance_api_pilot_302438/reparse_verification.json).
No pilot resubmission is needed.

The pilot uses the same baseline responses, goals and IDs **1, 1217, 1236**
(15 turns) as the failed local pilot. It requests **researchshort, eight CPUs,
8 GB RAM, one hour, no GPU**. Eight workers make concurrent API requests; one
writer appends and fsyncs completed records. User owns Slurm submission.

```bash
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
mkdir -p outputs/slurm
sbatch scripts/slurm/run_safedial_assistance_api_pilot.sbatch
```

Output: `outputs/assistance/gpt4o_mini_zephyr_pilot_v2/`.
Logs: `outputs/slurm/safedial-assistance-api-JOBID.{out,err}`.
The launcher uses the same `.env` / exported `OPENAI_API_KEY` and optional
`OPENAI_BASE_URL` as native judging. Existing environment values take precedence.
No key or endpoint credentials are written to the manifest. There is no extra
paid connectivity probe before the pilot's actual requests.

Preparation and request-size preflight require no API calls or GPU:

```bash
python -B scripts/judge_safedial_assistance_api.py \
  --run-dir outputs/safedial_baseline/zephyr_7b_beta_full \
  --output-dir outputs/assistance/gpt4o_mini_zephyr_pilot_v2 \
  --ids 1 1217 1236 --parallel 8 --preflight-only
```

The offline size check uses a conservative UTF-8 byte bound plus role-framing
allowance, not Zephyr tokenization or an exact API-token count. It reserves
the completion budget against GPT-4o-mini's 128,000-token context limit.
Actual token usage and returned model identity are recorded per API response.
See the [official model documentation](https://developers.openai.com/api/docs/models/gpt-4o-mini).

Review raw outputs, refusal/finish reasons and all 15 parsed labels before
expanding to full runs. Offline tests/preflight are not proof of label quality
or semantic label quality. The live request/format pilot plus audited reparse
is now complete; this is technical validation, not a human accuracy audit.

## Resume with normalized labels

**Submitted: job302464, RUNNING since24September15:49:14SGT. Do not submit a
duplicate.** Zephyr imported all10,029responses, corrected465format errors
offline, and completed assistance plus the saved LlamaGuard join with no extra
API calls. Labels:8,248NO/1,781YES; zero errors. CAT is nowrunning (84valid,
zero errors at15:49:48); the remainingfour sources follow. Original302453
finished with exit2 at15:35:44. Raw responses and previous successful labels
were verified unchanged. See [startup evidence](verification/assistance_normalization_20260924/startup_302464.json).

The normalized runner is ready. It accepts `NO.`, `**YES**`, quoted/fenced
labels, full-width characters and the numbered ASSIST forms while still
rejecting prose, negations and conflicting labels. GPT-4o-mini, the prompt,
24-token limit, extracted goals and eight workers are unchanged.

The user submitted the following from the DCGS root; the dependency allowed
it to start after the original suite finished:

```bash
sbatch --dependency=afterany:302453 \
  scripts/slurm/run_safedial_assistance_api_normalized_suite.sbatch
```

This CPU-only researchlong suite imports existing responses into separate
`gpt4o_mini_*_full_v2` directories, normalizes saved labels offline, then judges
missing responses across the same six sources. It preserves the original
`*_full_v1` artifacts and raw API text. No API calls are needed to repair label
formatting. `normalization_import.json` records source hashes and corrections.
The old runner remains unchanged so job302453 can finish with its source pins
intact; the new process uses protocol v3 with both runner files pinned.

At15:35SGT, a read-only audit of9,844 saved Zephyr responses recovered all458
format errors and preserved all9,386 existing labels. This is a bounded live
snapshot, not a claim that the full suite is complete. All42 assistance tests,
five mocked suite control-flow cases and shell syntax checks pass. A real
15-response pilot import also passed without API calls.
See [audit](verification/assistance_normalization_20260924/latest_prefix_audit.json)
and [suite checks](verification/assistance_normalization_20260924/suite_checks.json).

For subsequent interrupted normalized runs, submit the same normalized script
without the old-job dependency. Successful imported/new labels are skipped.
Use `--retry-errors` only after inspecting genuine remaining errors; formatting
corrections do not require it. Do not overlap submissions. Logs are under
`outputs/slurm/safedial-assistance-api-normalized-suite-JOBID.{out,err}`.

## Full suite submitted

The following records the original submission. Use the normalized resume
command above for subsequent work.

All six completed sources passed offline full-input preflight and exact
LlamaGuard scope/input matching on24September. There are **60,146 available
responses** across these sources. Full suite **302453** started at15:20:53 SGT on24September onbat, CPU-only
researchlong/eightworkers. At15:22SGT, Zephyr had1318saved responses:
1238parsed labels and80errors, of which80are clear labels with a
trailing period (for example `NO. [/ASSIST]`). This is a parser punctuation
bug, not evidence of API failure. Other five sources have not started.
The current suite will stop after Zephyr if these errors remain. Preserve all
responses for offline reparse; do not modify active runner/source hashes or
resubmit duplicate jobs. The normalized resume above repairs these saved
responses in separate outputs after the original job ends.
See [startup evidence](verification/assistance_suite_302453/startup.json).


```bash
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
mkdir -p outputs/slurm
sbatch scripts/slurm/run_safedial_assistance_api_full_suite.sbatch
```

This is one CPU-only researchlong job (eight CPUs,8GB,24h). It runs sources
**sequentially**, with eight concurrent API requests within each source, keeping
the suite at eight concurrent requests overall. It retains GPT-4o-mini, the
unchanged prompt/suffix and24-token limit, extracted v3 goals and corrected
strict parser. Each source writes a separate frozen result directory and joins
the matching saved LlamaGuard labels without rerunning LlamaGuard.

| Order | Source | Available turns | Assistance output under `outputs/assistance/` |
| --- | --- | ---: | --- |
| 1 | Zephyr | 10,029 | `gpt4o_mini_zephyr_full_v1` |
| 2 | CAT | 10,029 | `gpt4o_mini_cat_full_v1` |
| 3 | GPT-4o | 10,028 | `gpt4o_mini_gpt4o_full_v1` |
| 4 | SmoothLLM primary | 10,028 | `gpt4o_mini_smoothllm_primary_full_v1` |
| 5 | VDCGS-96 | 10,003 | `gpt4o_mini_vdcgs96_full_v1` |
| 6 | RDCGS-96 | 10,029 | `gpt4o_mini_rdcgs96_full_v1` |

SmoothLLM's separate recovery is excluded. DCR and both384-token sources are
still generating, and correctedTPO has a queued resume; none belong in this
frozen full suite yet. Frozen full inputs include2037dialogues/10029turn slots
per source, with existing source gaps explicitly retained.

The suite continues past expected exit2 only when every available response
has a successful assistance label and matching guard judgment. Source gaps
remain incomplete in the per-source aggregates. Real judge errors, missing
guard labels or infrastructure failures stop the suite for inspection.
Successful suite exit means all **available responses** are judged, not that
all source generations became complete or the metric measures harmfulness.

After inspecting genuine errors in a normalized run, an explicit retry is
available (format-only errors are repaired offline):

```bash
sbatch scripts/slurm/run_safedial_assistance_api_normalized_suite.sbatch --retry-errors
```

Do not run overlapping suite and per-source jobs against these directories.
Logs: `outputs/slurm/safedial-assistance-api-suite-JOBID.{out,err}`.
Evidence: [six-source preflight and guard matching](verification/safedial_assistance_api_full_20260924/preparation.json),
[offline suite control-flow checks](verification/safedial_assistance_api_full_20260924/suite_checks.json).

## Individual full runs

The general launcher is CPU-only, with eight workers and a 24-hour limit on
researchlong. It uses a fresh directory for each source and full scope:

```bash
sbatch scripts/slurm/run_safedial_assistance_api.sbatch \
  outputs/safedial_baseline/zephyr_7b_beta_full \
  outputs/assistance/gpt4o_mini_zephyr_full_v1 \
  --llamaguard-dir outputs/llamaguard/zephyr_7b_beta_full_v2
```

The optional guard join checks exact generation/history hashes and scope. Use
the matching full LlamaGuard directory; the 15-turn pilot cannot join directly
to a full-scope guard run. CAT, GPT-4o, SmoothLLM primary, VDCGS-96 and RDCGS-96
can use the same general launcher with their respective completed generation
directories and fresh assistance output paths. Do not freeze an ongoing DCR,
TPO or 384-token run as though it were a completed full run.

Repeated identical submissions skip successful judgments. Recorded errors are
not retried unless `--retry-errors` is explicitly added to the general launcher.
Up to three transport attempts are allowed for transient connection/timeout,
408/409/429 and server failures, with exponential backoff. Invalid labels and
refusals receive no automatic format retries. Each record includes response ID,
returned model, finish reason, raw output, usage, attempts and latency.

The output directory is locked and the request configuration, source code,
goals and frozen inputs are hashed. Changed configuration is rejected on resume.
A model-alias backend can evolve; returned model/fingerprint evidence is kept,
and the code does not claim a pinned OpenAI weight revision.

Missing source responses remain explicit, and exit 2 means incomplete coverage
or judge errors. All available responses can be judged while full coverage is
still incomplete. The combined score remains the previously documented
retrospective safety-plus-assistance score, not official SafeDialBench scoring.
