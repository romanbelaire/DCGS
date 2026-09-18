# Full-run preparation audit

The user requested inspection of original DCGS smokes and full-run commands
if acceptable. A40 jobs257599(VDCGS) and257600(RDCGS) passed five turns each.
Both `--require-gpu` validators were rerun successfully during this review.

`smoke_review.json` preserves the actual answers, parsed belief counts,
generation lengths and replay/GPU reports. Acceptance is for faithful
benchmark execution, not a claim of answer quality. All generation calls
used their96/128-token limits; answers contain cut-offs and placeholders,
and RDCGS turn5 follows an off-topic financial belief. Original critic
truncation occurred on turn5 in both methods. These settings remain intact.

The method subdirectories preserve smoke manifests and reports, prepared
full manifests, and exact launcher copies. Full manifests differ from smoke
manifests only in selected IDs and the declared terminal-failure continuation
policy. Runtime source hashes, checkpoints, seeds and model settings match.

Full preflight logs are CPU/tokenizer checks with synthetic beliefs, not
model outputs. They report context estimates separately from actual smoke
measurements. The launchers repeat preflight on the allocated node, execute
the benchmark, audit complete outputs with GPU evidence, and dry-run judging.
Infrastructure/integrity failures stop; terminal model outputs remain failed
while later independent turns continue. No extra blank-response retries.

Shell syntax checks passed for both launchers. No runtime code changed, so
the existing117-test result is prior evidence, not a new test run here.
No agent submitted or mutated a Slurm job or made a paid judge call.
