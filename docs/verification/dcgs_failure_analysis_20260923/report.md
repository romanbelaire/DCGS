# Completed SafeDialBench DCGS failure analysis

Scope: completed main-method v3 VDCGS/RDCGS runs with the 96-token belief-list
budget. This is an execution/failure analysis, not a harmful-response assessment.
The separate 384-token runs are outside this completed-run comparison.

## Outcome

| Run | Successful turns | Terminal turn failures | Complete dialogue exports | Final job |
| --- | ---: | ---: | ---: | --- |
| VDCGS96 | 10,003 / 10,029 | 26 (0.259%) | 2,011 / 2,037 | 300417: exit 2 |
| RDCGS96 | 10,029 / 10,029 | 0 | 2,037 / 2,037 | 296224: exit 1 |

VDCGS exhausted the dataset: zero unattempted turns. Its 26 failures occur in
26 different dialogues. The native answer export omits those incomplete dialogues,
even though their other successful turns remain in turns.jsonl. Keep coverage
explicit when comparing native scores with RDCGS.

## 25 VDCGS belief-list failures

Each failed turn has three recorded belief-generation attempts. Offline execution
of the exact `_parse_belief_candidates` and `_clean_candidate_text` methods against
all 75 saved outputs reproduces five `[SKIP]` entries every time. This replay
loads only the parser methods from the local Python AST; no model/API is used.

- All 75 attempts generated exactly 96 tokens, the configured maximum for the
  entire requested five-candidate list (not 96 tokens per candidate).
- 67 outputs contain only the numbered marker `1.`.
- Five outputs contain `1.` followed by malformed `12.` rather than `2.`.
- Three outputs reach `2.` but have no text for that second item.
- Seven outputs also repeat the first number (`1. 1.`). This is an observed
  formatting defect, not established as an independent cause of rejection.

The parser accepts a full list or a partial consecutive list containing at least
two items. A single substantial candidate is discarded; absent slots are padded
with `[SKIP]`. The original policy retries twice, then terminates the turn when
all slots are `[SKIP]`. The wrapper records `beliefs_exhausted` and continues to
later benchmark turns under fixed gold history.

Thus these failures reflect a tight list-generation budget interacting with
list-format/parser requirements. They are not blank model responses, confirmed
safety refusals, GPU out-of-memory errors, or an absence of any meaningful text.
The evidence supports investigating a larger belief budget or a separately
versioned parser/single-candidate policy; it does not establish that either would
fix every case or improve safety. Changing those settings changes the experiment.

Example: dialogue 155, turn 1, produces a substantive first Insight/Instruction
on each attempt but never a valid second list item before the token cap.
See summary.json for every failed belief turn and all attempt diagnostics.

## One VDCGS low-level critic overflow

Dialogue 987, turn 6 (zero-based index 5), reaches response-candidate scoring.
The LL critic receives a sequence of 8,232 tokens, exceeding its 8,192-token limit
by 40. `tokenize_action_span` explicitly refuses truncation and raises
`LLContextLengthError`, recorded as `ll_context_overflow`.

Generation already occurred; the failure is scoring the candidates, not an OOM
or exhausted belief retries. The audit also shows selection of a truncated
belief and an off-topic donation response pool, but those quality observations
are distinct from the mechanical overflow and are not assigned a safety score.

## RDCGS final validation failure and historical runtime records

RDCGS generated all 10,029 turns. Its final `--require-gpu` validator raises
`Successful model work lacks GPU memory evidence`.

| Method / invocation | Successful turns | Loading evidence | Runtime / GPU peak |
| --- | ---: | --- | --- |
| VDCGS original a603a48... | 8,974 | cuda:0, bfloat16 | Missing |
| VDCGS resume 9ce095f... | 1,029 | cuda:0, bfloat16 | Present, 19,454,815,232 bytes |
| RDCGS original 16cce8a... | 7,762 | cuda:0, bfloat16 | Missing |
| RDCGS resume 4e99f73... | 2,267 | cuda:0, bfloat16 | Present, 19,454,344,704 bytes |

Original jobs 258754/258755 ended by Slurm TIMEOUT after 48 hours. Runtime rows
are written only in the runner's finalization block; the original invocations
have loading/output records but no final runtime rows. A later resume cannot
supply the earlier invocation's historical peak. This is a telemetry/validation
gap, not evidence that generation failed or used the CPU.

VDCGS also has this historical evidence gap, but its final exit 2 is the expected
incomplete-coverage outcome. Its isolated final audit reports integrity and
policy parity true, with 26 terminal failures and zero remaining turns.
RDCGS's validator reached the GPU-evidence check after the integrity/policy/native
export checks. Do not regenerate 10,029 successful responses solely to clear
Slurm's FAILED label or fabricate missing historical GPU measurements.

## Earlier VDCGS resume bug: resolved

Job 296226 stopped during saved-turn replay with
`Original-policy selection/audit mismatch`. Replay returned a shared `texts`
list; upstream all-SKIP retries replaced entries in it and mutated audit evidence
in memory. On-disk output was preserved. The separate replay-isolation wrapper
copies the trace before validation; resumed job 300417 passed integrity and
policy replay and completed generation. This historical bug is distinct from
the 26 terminal turn failures above.

## Separate quality limitation

The final VDCGS audit reports high-level critic truncation on 3,369 successful
turns (6,759 / 20,898 candidate contexts), at the existing 1,500-token critic
limit. This can affect what the critic sees, but it did not itself terminate
those turns and is not proof of unsafe answers. Do not confuse it with the
single hard overflow at the LL critic's 8,192-token limit.

## Verification and source pointers

Read-only analysis of status, failure ledgers, raw failed generation tokens/text,
native exports, model-loading/runtime journals, final validator output and Slurm
accounting. Exact saved-parser replay: 75 / 75 all-SKIP outcomes reproduced.
No inference, API calls, job changes, or source-pinned runtime edits.

- `outputs/safedial_dcgs/zephyr_vdcgs_main_wildjailbreak_full_v3/{failures,turns,answers,runtime,model_loading}.jsonl`
- Same files for `zephyr_rdcgs_main_wildjailbreak_full_v3` (no failure ledger).
- `src/agents/high_level_agent.py`: `_parse_belief_candidates`.
- `src/main.py`: `MAX_BELIEF_ATTEMPTS` and all-SKIP handling.
- `src/value/ll_token_critic.py`: `tokenize_action_span`.
- `scripts/validate_safedial_dcgs_wildjailbreak.py`: per-invocation GPU evidence.
- `scripts/run_safedial_dcgs_wildjailbreak.py`: runtime finalization.
- `scripts/safedial_dcgs_replay_isolation.py`: prior replay-bug workaround.
