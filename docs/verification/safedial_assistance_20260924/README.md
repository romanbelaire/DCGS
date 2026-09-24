# Assistance runner verification — 24 September 2026

- 21 new assistance tests and 69 total related offline tests pass.
- Original local DCGS rubric is checked against the actual function AST on
  untruncated inputs; full-input preservation is checked separately.
- Full real-data cached-tokenizer preflight passes for 60,146 available responses
  across Zephyr, CAT, GPT-4o, SmoothLLM primary, VDCGS-96 and RDCGS-96.
- Maximum prompt lengths: Zephyr 1,314; CAT 1,256; GPT-4o 1,558;
  SmoothLLM 1,295; VDCGS-96 818; RDCGS-96 825. Limit 8,192; zero clipping/overflow.
- Missing/error generation stays explicit: GPT-4o 1 missing, SmoothLLM 1 error,
  VDCGS-96 26 missing. These are not labeled NO or excluded silently.
- Real GPT-4o and RDCGS-96 saved LlamaGuard sources match the assistance source
  hashes exactly. With no assistance labels, combined scores correctly remain
  incomplete and percentages null. Fixture joins cover all four label pairs
  and reject changed response/scope inputs.
- Frozen goal provenance validates all 2,037 records, including user-approved
  dialogue 1217. Existing goal artifacts, extractor, guard runner and original
  DCGS reward source hashes are unchanged (see real_preflights.json).
- Python compilation, launcher and guide shell syntax, local links, whitespace pass.
- No model inference, paid API calls, downloads, or Slurm submissions performed.
  Actual label quality, parse success rate, runtime and GPU memory await the pilot.

Preparation directories are under /tmp/safedial-assistance-*-20260924;
production judge directories were not created. Reports contain no fabricated
production labels. tests.log and related_tests.log contain only fixture results.
