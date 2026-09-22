# DCR-main review for SafeDialBench

Inspected 2026-09-21; inference-file follow-up 2026-09-22. Source folder:
`models/dcr_sft_lora_171151/DCR-main`. No inference/training source was changed,
no upstream module imported, and no GPU/API/Slurm work was started.

The user confirmed that `dcr_sft_lora_171151` was trained with this exact
`training/SFT_lora.py`. This supersedes the earlier assumption that the uploaded
`chat_template.jinja` was the training serializer: this script uses the external
helper, not `tokenizer.apply_chat_template`. The helper implementation is still
missing, so equivalence to any tested template cannot be established.

## Recommendation

On 2026-09-22 the user explicitly chose the supplied inference file's format
for SafeDialBench despite unverified training parity. The missing helper is
therefore not a blocker for this chosen experiment. The prepared full launcher
and its qualifications are documented in [the DCR guide](SAFEDIALBENCH_DCR.md).
The reproduction questions below remain distinct from this authorization.

Use this folder to establish the trained model and preprocessing contract.
The supplied workflow is contrastive base refinement followed by LoRA SFT,
then ordinary causal-LM generation. Keep the existing SafeDialBench turn loop,
gold-assistant history, saved-response format and independent judge. Load the
correct final weights and reproduce the actual training/inference serialization.
There is no need to run contrastive optimization, classifiers, or the paper's
other benchmark evaluators during each SafeDialBench turn.

The folder is incomplete for exact reproduction. It does not establish that
the current stock-Qwen + uploaded LoRA checkpoint contains contrastive refinement,
nor that the uploaded uppercase ROLE template matches this source's training.
Do not change the base/adapter pairing or declare a new template correct from
this folder alone. Current smoke artifacts remain valid records of the tested
configurations, but do not establish reproduction of the complete DCR workflow.

## Evidence

| Finding | Source | Implication |
| --- | --- | --- |
| Contrastive stage loads a complete causal LM, optimizes it and saves the underlying model | `base_model_contrastive.py:263`, `:286`, `:569` | Refinement resides in base weights; it is not an inference-time prompt transformation |
| SFT and inference use the contrastive output as their base | `contrastive+lora+evaluate_pipeline.py:34`, `:40`; Qwen experiment `experiment/contrastive+lora+evaluate+differentdata_circle.py:62`, `:69` | Reproduction needs this refined base plus its matching SFT LoRA, or an already merged final model |
| SFT saves only LoRA weights and tokenizer | `training/SFT_lora.py:199` | The adapter does not automatically include earlier full-base updates |
| Uploaded params and adapter metadata both name stock Qwen; SFT data path is toxic500_xstest | `../params.json:2`, `../adapter_config.json`; Qwen pipeline uses toxic500_xstest at stage 1 and alpaca_20k_safety1k at stage 2 | Metadata differs from the supplied two-stage recipe; exact training history is unresolved, not proof of a broken adapter |
| Qwen SFT calls an external `Qwen2_chat_template` helper | `training/SFT_lora.py:15`, `:122` | Saving a tokenizer template does not prove it was used to construct training examples |
| System text is `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`; user text is instruction + newline + input | `training/SFT_lora.py:123` | Our original smoke had no system message; standard-Qwen smoke used a different system message; neither proves exact source-format parity |
| `AttnLRP_Analysis/utils.py` remains absent; a `vllm_inference_lora.py` drop-in was supplied on September 22 | Folder inventory and new inference source | Supplied inference serialization is now known; exact training serialization and original inference provenance remain unresolved |
| Outer `add_eos_token=False` is shadowed by inner `tokenize(..., add_eos_token=True)` and the caller supplies no override | `training/SFT_lora.py:110`, `:142` | This source appends EOS; the params flag alone does not establish missing EOS training |

The actual helper implementation may or may not equal standard Qwen ChatML;
its function name is insufficient evidence. The original uploaded template was
previously user-confirmed, but the user now confirms this exact script; its
external helper is the authoritative serializer and remains missing. Likewise, standard-base metadata is not
proof that a separate refined base was intended for this particular adapter.

## Newly supplied inference file (2026-09-22)

`DCR-main/vllm_inference_lora.py` supplies the expected base/LoRA/prompt-file
CLI. It loads the tokenizer from `--model_path`, overrides its chat template,
and generates with vLLM plus `LoRARequest`. Defaults are temperature 0 and
512 new tokens. There are no explicit stop strings or stop-token IDs; effective
stopping still depends on the installed vLLM version and model configuration.

Its `CHAT_TEMPLATE` is byte-for-byte equal to the saved adapter
`chat_template.jinja`: uppercase role names, colon-space, newline after each
message, and an `ASSISTANT: ` generation prefix. It supplies only a user message,
without a system message. Nonempty input is joined to instruction with a newline;
empty input adds no newline to the user content. This is the template already
used by the original diagnostic arm, although that arm preserves multi-turn
SafeDial gold history and uses a 1,024-token cap and a different inference backend.

The docstring calls this a “Drop-in” and asserts that formatting matches
`training/SFT_lora.py`. That assertion is not verified: the confirmed training
script instead passes system/user/assistant messages to the missing external
`Qwen2_chat_template`, and always joins instruction and input with a newline.
Even if the helper used identical role delimiters, the supplied message contents
would differ. The helper could also transform or omit messages; its actual
behavior remains unknown. This file therefore resolves the behavior of this
inference implementation, but does not establish training parity or the original
inference script's provenance. Recover the actual training copy of
`AttnLRP_Analysis/utils.py` next. Base-refinement provenance remains a separate gap.

Verification: parsed source with AST without importing or running upstream code;
compared the literal template to the saved Jinja file; checked existing runner
serialization. Source hash and findings are recorded in
`verification/dcr_inference_source_review_20260922.json`. No inference, model,
runner, or scheduler changes were made.

## How to integrate once provenance is established

1. The user confirmed the exact training script. Recover its
   `Qwen2_chat_template` implementation from `AttnLRP_Analysis/utils.py` and
   establish whether the newly supplied inference drop-in is the exact version
   used to evaluate this checkpoint.
   Inspect a rendered training example and its token IDs, including the final
   assistant terminator/EOS. Recover training data schema or representative
   preprocessing evidence to establish how instruction/input were assembled.
2. If this adapter was trained on stock Qwen, keep that pairing and describe it
   as the supplied SFT-only model unless other evidence establishes refinement.
   For complete two-stage DCR, obtain the refined base and the adapter trained
   on that base, or a merged final checkpoint. Do not combine this adapter with
   an arbitrary refined base, Qwen-Instruct or Zephyr. Source code is not a
   substitute for trained weights, and fresh training is a separate task.
3. Create a separate versioned runner/config once these details are known.
   The current `prepare_safedial_dcr.py:10`, `:27`, `:77` deliberately pins stock
   Qwen and its hash; a refined checkpoint needs a new artifact specification
   and checks. Pin both stages, tokenizer, template/helper, system message,
   stopping rules, source hashes and dataset. Keep original run manifests intact.
4. Preserve SafeDialBench reference history, append current user, serialize with
   the recovered helper/template and generate only the next assistant response.
   Any adaptation from single-turn instruction/input to multi-turn messages
   must be explicit. Keep benchmark greedy decoding/1,024-token cap for a
   matched diagnostic; record any intentional difference from upstream defaults.
5. CPU-check exact formatting/tokenization, missing answer/future leakage,
   artifact compatibility and context sizes. Then prepare a fresh IDs1,2,3 smoke,
   retaining full generated IDs, empty outputs and errors. User submits Slurm.
   Compare to the four existing arms without selecting a format on cap rate alone.

## Limits and verification

- Read all relevant training, refinement and pipeline paths; inspected adapter
  metadata and compared existing runner model/template loading.
- Ran an isolated preprocessing fixture extracted with AST. It verifies the
  exact system/user/assistant message list, full-token labels, and appended EOS
  despite the outer false flag. The missing formatter was a capturing stub;
  this does not validate its serialization. Evidence and source hashes:
  `verification/dcr_source_review_20260921/inspection.json`.
- Pipeline files contain placeholder paths, hard-coded GPU choices, absent
  dependencies and commented-out stages. They should not be executed as-is.
- No causal explanation of repetition is established by this review. Artifact
  provenance, template parity and model quality are distinct questions.
