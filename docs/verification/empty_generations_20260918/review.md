# Empty-generation diagnosis — 2026-09-18

VDCGS256920 and RDCGS256921 failed on newline repetition. Each failed result
contains exactly1024 newline characters and1024 completion tokens, hit the
1024-token cap, and has no input truncation or backend error. Inputs were
1444 and1015 tokens respectively. The validator rejects message.strip() == ""
before response-tag parsing or token-critic scoring. One invalid candidate
stops the turn and full job with exit2; successful earlier candidates remain
saved but the required five-candidate selection is incomplete.

This is an existing degeneration pattern: VDCGS has2 generations with >=10
trailing newlines among177 saved generations; RDCGS has11 among706 (counts
include each fatal generation). Prior nonempty candidates can pass validation
while spending most of their token budget on trailing newlines.

DCGS uses raw repository prompts plus separately tokenized [RESPONSE] prefill,
not the actor chat template. CPU retokenization exactly matches saved prompt
lengths. This formatting is an implementation choice and a potential hypothesis
for a controlled experiment, not a demonstrated root cause. Generation has no
anti-repetition constraint (repetition_penalty1, no_repeat_ngram_size0); the
custom suppression only masks the four Example/example/Examples/examples
word IDs16693,2757,26268,9254. It does not force or suppress newline token13.
The saved trace establishes newline repetition but not why the model entered it.

TPO256711 failed at dialogue28/index2/event22, loss-feedback round1/slot0.
It generated1 token in0.155s; decoded/stripped message is empty, input1867,
no truncation, cap2048 not reached. Immediate EOS is strongly supported by
normal generation stopping with EOS2 enabled and no max_time/stop_strings or
minimum-new-token guard. Raw token IDs and unstripped decoded text are not
saved, so exact token identity cannot be proven from artifacts. This failure
precedes update parsing and is unrelated to the earlier IMPROVED_VARIABLE bug.
CPU chat-template reconstruction exactly matches the1867-token prompt.

SmoothLLM dialogue344/index4/candidate0 has two saved failed attempts with
same seed551667940,1024 completion tokens and empty post-strip text; input3489,
no truncation. Compatible with whitespace repetition, but unlike DCGS its
backend strips before saving and raw token IDs are absent. Exact whitespace
characters cannot be recovered. The repeated failure demonstrates that an
unchanged retry did not fix that candidate.

Verification: all96 pinned source hashes match for each DCGS run,8 for TPO,
4 for SmoothLLM. Exact DCGS/TPO validators replay their saved errors on CPU.
No evidence of context overflow, OOM, or response parsing discarding substantive
DCGS text. No GPU/model/API calls, Slurm mutations, or runtime edits performed.

Next diagnostic improvement: retain raw generated token IDs, decoded text before
stripping, and stop reason. Recovery would need explicit bounded retries with
recorded new seeds and preserved failed attempts, or a separately validated
sampling/prompt change. Increasing the token cap alone does not address either
observed failure mode. Unchanged seeded retries are not a verified remedy.
