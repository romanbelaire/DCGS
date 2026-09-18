# Empty-generation recovery proposal

## Finding: removing whitespace trimming does not fix the saved failures

The DCGS backend already saves decoded output without stripping. Its validator
uses message.strip() only to test for substantive content; it does not mutate
the saved string. The failed messages are newline-only, not valid answers with
extra trailing whitespace. Calling the exact response parser directly on the
raw failed strings, bypassing that validator, still returns an empty string.

Compared parsing [RESPONSE]+raw with [RESPONSE]+raw.strip() for every saved
DCGS response generation:147 VDCGS and504 RDCGS,651 total. All parsed strings
are identical. This includes both failures and12 response candidates with
>=10 trailing newlines. Useful content plus trailing whitespace already works.
Evidence:whitespace_parser_check.json. Exact parser method was AST-extracted
and executed offline, without editing source or loading any models.

A synthetic closed whitespace-only tag pair exposes an adjacent parser weakness:
[RESPONSE]\n\n[/RESPONSE] falls back to returning the raw tag string. That did
not cause these failures. Any recovery implementation should explicitly reject
content that consists only of formatting tags/whitespace; relaxing validation
would make this loophole worse.

## Recommended first recovery policy for sampled DCGS and TPO

1. Keep the substantive-output check. Save raw decoded text and output token IDs
   separately from parsed/trimmed text; record termination reason and costs.
2. Retry only the invalid generation call, with at most two extra attempts.
   Keep its prompt, chosen intent, temperature, top-p and token budget fixed;
   derive distinct reproducible seeds from original seed plus retry index.
   Do not rerun completed intent selection, prior candidates, or reward scores.
3. Record every attempt and its seed/tokens/latency; accept the first valid
   output rather than the highest-scoring retry. Preserve the nominal five
   DCGS response slots and the TPO candidate/iteration schedule. Extra attempts
   still change the effective method/cost and must be declared in the manifest.
4. If all attempts fail, retain an explicit unresolved failure. Do not turn
   whitespace into an answer, fabricate a refusal, or silently drop the slot.
5. Version the recovery policy and prepare a new continuation directory with
   prior artifacts preserved. The existing request/source-hash guards correctly
   prevent silent mutation of a benchmark run. --retry-errors alone currently
   repeats the same seed and is not this proposed recovery policy.

This is a proposal, not a demonstrated GPU fix. Saved-event replay can verify
preservation/accounting, but only model execution can show whether the new
seeds produce usable output for the actual failed cases.

## SmoothLLM requires a different test

Its manifest has temperature0.0 and defense do_sample=false; the common backend
therefore uses greedy decoding. Merely changing a generation seed does not
change its next-token selection. Regenerating the perturbation would change
which copy is being voted on and is not a neutral retry.

For this arm, test an explicitly versioned generation guard on the identical
perturbed prompt: limit consecutive whitespace-only continuation tokens and
exclude EOS while no substantive generated content exists. Base detection on
the generated continuation, not input history/prefill; handle tokenization of
spaces/newlines and other invisible/special tokens. Normal internal whitespace
must remain allowed. Such a guard changes decoding, needs CPU/tokenizer tests
and a controlled GPU smoke, and cannot be claimed to guarantee a useful answer.
Keep the existing active run's source files unchanged while it finishes.

## Alternatives and limits

- min_new_tokens delays EOS; it does not require a non-whitespace token and
  therefore is not a fix for the DCGS newline loops by itself.
- A global repetition_penalty or no_repeat_ngram_size changes ordinary answers
  too. Prefer a narrowly scoped recovery policy initially; compare a whitespace
  guard as a separate experiment if retry failures persist.
- DCGS uses the original raw prompt/prefill route. A chat-template ablation is
  worth a separate controlled experiment, but no evidence yet establishes that
  formatting is the cause or that changing it preserves the intended method.
- More tokens or accepting a blank candidate does not recover missing content.

## Validation before continuation

Offline: exact real-failure fixtures; substantive answers with whitespace;
tag-only rejection; attempts exhausted; seed determinism; all-attempt costs;
interruption/resume at every retry; unchanged good calls and no-op replay.
GPU: reproduce failed prompts under original settings with raw token logging,
then try bounded retries for DCGS/TPO and a separately labeled guard for greedy
SmoothLLM. Inspect meaningful output, selected candidates, and saved audit.
No GPU/model/API execution or production-source edit was performed here.

Reference: https://huggingface.co/docs/transformers/main_classes/text_generation
(Hugging Face documents do_sample/greedy, min_new_tokens, repetition controls.)
