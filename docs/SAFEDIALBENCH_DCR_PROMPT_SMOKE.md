# DCR prompt-format smoke comparison

This diagnostic tests whether the language switching and repeated dialogue seen
in DCR smoke 271289 recur on two further dialogues, and how outputs change when
the training chat template is omitted. It does not establish that the template
caused the original behaviour.

**Completed smoke results (2026-09-21):** training-more job 271383 and literal
Zephyr job 271399 passed their technical audits. Raw job 271384 ran all15
turns but failed output validation on five decoded-empty responses. Across
matched IDs1,2,3, training/raw/Zephyr cap hits were14/6/8; extra-role replies
were11/0/4, with raw also showing plain-text repetition. See the
[comparison and complete responses](verification/dcr_prompt_comparison_20260921/README.md).
No quality judging or automatic retries were performed.

## Selected cases and controls

| Mode | IDs | Turns | Purpose |
| --- | --- | --- | --- |
| `training` | 2, 3 | 10 | Two additional dialogues using the uploaded training template |
| `raw` | 1, 2, 3 | 15 | Same cases, including the original domestic-violence dialogue, without chat formatting |
| Literal Zephyr format | 1, 2, 3 | 15 | Same cases with Zephyr-style role markers and `</s>` boundaries; separate launcher below |

ID 2 begins with a question about school violence (Ethics). ID 3 begins with
a question about Black people (Aggression). They were selected as the next two
dataset IDs before examining any DCR outputs for them. The templated comparison
for ID 1 is the preserved output of job 271289; do not regenerate it silently.

Both modes use the same pinned standard Qwen2.5-1.5B, uploaded LoRA weights,
tokenizer/vocabulary, EOS/padding token, BF16, greedy decoding, 1,024-new-token
cap, seed 0 and reference-assistant history. No system message or new stop
strings are added. Both modes derive per-turn seeds from the original DCR
model ID, despite using distinct output model IDs. Their selection does not
affect per-turn seeds. There is no output trimming or repetition suppression.

`training` uses the uploaded `chat_template.jinja`:

```text
USER: <previous user message>
ASSISTANT: <reference assistant reply>
USER: <current user message>
ASSISTANT:<space>
```

Here `<space>` denotes the literal trailing space in the assistant cue.

`raw` concatenates the **same message contents in order**, separated by two
newlines, with no added role labels or final assistant cue:

```text
<previous user message>

<reference assistant reply>

<current user message>
```

At the first turn, raw input is just the current user message. No
`apply_chat_template` call is made for raw benchmark prompts; tokenization
uses `add_special_tokens=False` in both modes. Uploaded special-token
definitions and EOS stopping are retained. Text already inside dataset
messages is preserved verbatim, even if it contains role-like strings.

The fixed benign adapter enabled/disabled-logit verification fixture still
uses the training template in both modes; it is separate from benchmark
generation. Raw formatting differs from the adapter's confirmed training
format and removes speaker boundaries, so a raw-text result is diagnostic,
not automatically the correct production inference protocol.

## User submission

From the DCGS root:

```bash
mkdir -p outputs/slurm
sbatch --job-name=dcr-training-more scripts/slurm/run_safedial_dcr_prompt_smoke.sbatch training
sbatch --job-name=dcr-without-template scripts/slurm/run_safedial_dcr_prompt_smoke.sbatch raw
```

Each job requests one L40, 32 GB host memory, four CPUs and one hour. The jobs
can run independently and write to different directories:

- Training: `outputs/safedial_baseline/dcr_qwen_1.5b_prompt_training_ids2-3_smoke_v1`
- Raw: `outputs/safedial_baseline/dcr_qwen_1.5b_prompt_raw_ids1-2-3_smoke_v1`

Logs: `outputs/slurm/safedial-dcr-prompt-smoke-<JOB_ID>.{out,err}`.
The launcher validates all selected prompts, generates responses, audits
the saved outputs and runs judge compatibility checks without paid API calls.
The user owns submission; the implementation agent has not launched jobs.

## Implementation and review

### Additional Zephyr-style diagnostic

The user also requested the baseline's Zephyr format. The separate
`run_safedial_dcr_zephyr_prompt.py` uses exactly this text serialization:

```text
<|user|>
<previous user message></s>
<|assistant|>
<reference assistant reply></s>
<|user|>
<current user message></s>
<|assistant|>
```

It retains Qwen's tokenizer, adapter, vocabulary and EOS stopping. Under the
uploaded Qwen tokenizer, `<|user|>` and `<|assistant|>` each encode to five
ordinary tokens, and `</s>` encodes to three ordinary tokens. They are **not**
registered special tokens. Qwen's actual EOS remains `<|endoftext|>` (151643).
No vocabulary changes, embedding resizing, added stop sequences or output
trimming are performed. Therefore this tests literal Zephyr-style formatting;
it does not give Qwen Zephyr's trained boundary-token semantics. It also differs
from the adapter's confirmed training format.

Submit this additional 15-turn comparison:

```bash
sbatch scripts/slurm/run_safedial_dcr_zephyr_smoke.sbatch
```

The job requests one L40/32 GB host memory/four CPUs/one hour, with the same
greedy 1,024-new-token cap, BF16, reference history and per-turn seeds. Output:
`outputs/safedial_baseline/dcr_qwen_1.5b_prompt_zephyr_ids1-2-3_smoke_v1`.
Logs: `outputs/slurm/safedial-dcr-zephyr-smoke-<JOB_ID>.{out,err}`.
The original training/raw scripts and already prepared run identities remain
unchanged. The new wrapper loads the shared diagnostic into a private module
namespace and pins both its own source and its dependencies. Its audits also
report generated Zephyr-role markers and literal `</s>` occurrences. There are
no paid judge calls. Evidence: `verification/dcr_zephyr_prompt_20260921/`.

### Shared validation

The isolated `scripts/run_safedial_dcr_prompt_ablation.py` preserves the
original runner and smoke artifacts. It records explicit prompt-format
identities, exact rendered prompts and generated token IDs, and rejects
cross-mode output reuse. Resume, artifact checks, nonblocking output locks
and failed-attempt preservation follow the original DCR runner.

CPU checks verify original-versus-new training-mode rendering and tiny-Qwen
generation parity, no template call in raw generation, no current-reference
or future-turn leakage, paired seeds, output audit, tamper rejection and
unchanged no-op resume. Real-tokenizer preflight covers all10 training and
15raw turns. Maximum prompts: 1,458 tokens training and 1,486 raw; no overflows.
These checks are not GPU results for the new experiment.

After completion, inspect each mode's `validation.json` and `turns.jsonl`.
Validation reports per-turn cap hits, terminal EOS, CJK character counts and
generated role-label counts. Read the responses too: repeating prose or
numbered lists may contain no generated role markers. Compare corresponding
dialogue/turn IDs, and retain the first raw response even if empty or poor.
Structural success is separate from answer quality; no automatic repair is
applied. Do not interpret this small selected smoke as a benchmark-wide result.

Audit without inference:

```bash
.venv/bin/python -B scripts/run_safedial_dcr_prompt_ablation.py \
  --output-dir outputs/safedial_baseline/dcr_qwen_1.5b_prompt_raw_ids1-2-3_smoke_v1 \
  --audit-only --require-gpu
```
# Standard Qwen template smoke

The fourth arm uses the exact `chat_template` from the official
`Qwen/Qwen2.5-1.5B` tokenizer configuration at the same pinned base revision
`8faed761d45a263340a0528343f099c05c9a4323`. The configuration is vendored in
`configs/safedial/qwen_standard_template/tokenizer_config.json` and hash-checked.
Source: https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/8faed761d45a263340a0528343f099c05c9a4323/tokenizer_config.json

Run all IDs 1,2,3 (15 turns), matched to training/raw/literal-Zephyr smokes:

```bash
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
sbatch scripts/slurm/run_safedial_dcr_qwen_smoke.sbatch
```

User submission only. The launcher requests one L40, 32 GB host RAM, four CPUs
and one hour; it runs CPU preflight, inference, GPU/output audit and judge
dry-run. It makes no paid judge calls. Output is separate:
`outputs/safedial_baseline/dcr_qwen_1.5b_prompt_qwen_standard_ids1-2-3_smoke_v1`.

This changes the benchmark prompt to standard Qwen formatting, including its
automatic `You are a helpful assistant.` system message, `<|im_start|>` /
`<|im_end|>` boundaries and final assistant cue. Thus it tests the complete
default template, not just the boundary tokens. The uploaded tokenizer,
standard base + DCR LoRA, reference history, paired seeds, BF16, greedy decoding
and 1,024-token limit remain as in the previous arms. The adapter activation
probe still uses the uploaded training template, as in all previous arms.

Stopping remains the base/uploaded tokenizer's `<|endoftext|>` (151643).
`<|im_end|>` (151645) is a registered special token but is NOT added as a stop
token. This is a prompt-only comparison, not a switch to Qwen-Instruct generation
defaults. Both marker tokens are removed from decoded text, so the new audit
counts them in raw completion IDs. `raw_generations.jsonl` saves generated IDs
and decoded text before empty-output validation; a decoded-empty reply still
counts as a failure and causes strict validation to fail, with no automatic retry.

CPU checks, prepared manifest and all 15 rendered prompts are under
`docs/verification/dcr_qwen_prompt_20260921/`. GPU job271466 completed successfully; all15replies hit the token cap.
See [the four-template comparison](verification/dcr_qwen_comparison_20260921/README.md). Compare the resulting replies, errors, cap hits,
EOS endings and role continuations with the prior 15-turn table in
`docs/verification/dcr_prompt_comparison_20260921/README.md`. These smoke metrics
do not establish benchmark-wide safety or quality. Previous results are retained.
