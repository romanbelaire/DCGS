# SafeDial DCGS: 384-token belief lists

The separate `scripts/run_safedial_dcgs_belief384.py` entrypoint sets
`freeform_max_new_tokens=384` for VDCGS and RDCGS. This is the maximum for each
generated candidate list, not for each belief. The existing 96-token runners,
configuration files, and output directories are preserved.

The launcher requires an explicit context choice: `history` gives the LL
generator supplied prior gold turns, the current user message, and the selected
belief; `belief-only` gives it the selected belief. Current reference answers
and future turns are excluded. Both settings use the same existing critics,
1500-token HL critic limit, 8192-token LL critic limit, LL pool budget 640,
fallback budget 128, and original candidate selection and retry policies.

## Submit

Each job requests one L40S, 64 GB RAM, eight CPUs, and 48 hours in researchlong.
At the 2026-09-21 08:56 SGT read, lotus and lunar each had two unallocated L40S;
all L40 GPUs on lagoon and lexicon were allocated. Availability can change.

```bash
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
sbatch --job-name=safedial-vdcgs-b384-full scripts/slurm/run_safedial_dcgs_belief384_full.sbatch vdcgs belief-only
sbatch --job-name=safedial-rdcgs-b384-full scripts/slurm/run_safedial_dcgs_belief384_full.sbatch rdcgs belief-only
```

The user selected `belief-only` to retain the existing full runs' LL context
setting for the 96-versus-384 comparison. For an actual L40 allocation, add
`--gres=gpu:l40:1` before
the script path. Submit only one desired context variant per method.

Outputs are `outputs/safedial_dcgs/zephyr_{method}_main_{context}_belief384_full_v1`.
Logs are `outputs/slurm/safedial-dcgs-belief384-full-{job_id}.{out,err}`.
Both jobs cover 2037 dialogues / 10029 turns. They first perform a complete
tokenizer-only preflight, then generate with record-and-continue for terminal
model-output failures. Infrastructure and integrity errors stop execution.
Failed turns remain in the ledger; only complete dialogues enter the export.
Completion with failed turns returns 2. Judging at the end is a dry-run only.

Each context has a separate model/protocol identity, effective configuration,
and source hashes. Reuse of incompatible output is rejected. A prepared recovery
directory can be supplied as the third launcher argument. Keep the new runner
and launcher unchanged while these jobs run.

Seed 0 is retained, but the existing seed derivation includes model identity,
so these runs do not share exact per-turn random seeds with the 96-token runs.
A comparison with the current belief-only runs also changes LL context when
`history` is selected; it cannot isolate the effect of the belief budget.

## Verification

51 DCGS tests pass, including actual 384-token request construction, context
isolation, pool/fallback history, replay, byte-identical no-op resume, and
tampered-budget rejection. See `docs/verification/belief384_20260921` for
pinned-tokenizer one-dialogue preflights and manifests. Full tokenizer scans
run in each submitted job. No 384-token GPU generation or quality judging was
performed during preparation.
