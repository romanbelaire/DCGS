# Active SafeDial scripts

Superseded experiments and one-time migrations were archived on 2026-09-18.
Use this directory for current work. See the [audit archive](../archive/scripts_cleanup_20260918/README.md)
for exact old files, hashes, run references, and historical tests.

| Task | Entry point | Validation / launcher |
| --- | --- | --- |
| Original WildJailbreak DCGS | `run_safedial_dcgs_wildjailbreak.py --method vdcgs` or `rdcgs` | `validate_safedial_dcgs_wildjailbreak.py`; `slurm/run_safedial_{vdcgs,rdcgs}_wildjailbreak_original_{smoke,full}.sbatch` |
| TPO, current full run | `run_safedial_tpo_full.py` | `validate_safedial_tpo_full.py`; `slurm/run_safedial_tpo_retry_full.sbatch` (v8) |
| TPO, upstream candidate handling (new, GPU smoke pending) | `run_safedial_tpo_upstream_full.py` | `validate_safedial_tpo_upstream.py`; `slurm/run_safedial_tpo_upstream_smoke.sbatch`; [policy guide](../docs/SAFEDIALBENCH_TPO_UPSTREAM_HANDLING.md) |
| SmoothLLM, resumable full run | `run_safedial_smoothllm_turn_resume.py` | `validate_safedial_smoothllm_turn_resume.py`; `slurm/run_safedial_smoothllm_turn_resume_full.sbatch` |
| Zephyr baseline / CAT | `run_safedial_baseline.py` | `validate_safedial_generation.py`; Zephyr/CAT launchers in `slurm/` |
| GPT-4o baseline | `run_safedial_api.py` | Same entry point with `--validate-output`; GPT-4o launchers in `slurm/` |
| Judging | `judge_safedial.py` | `--dry-run` checks inputs without paid calls; dedicated judge launchers in `slurm/` |
| Human adjudication export | `export_safedial_adjudication.py` | See `docs/SAFEDIALBENCH_HUMAN_ADJUDICATION.md` |
| Artifact preparation | `prepare_safedial_cat.py`, `prepare_safedial_tpo.py` | `--offline` validates existing assets |
| API connection check | `test_openai_connection.py` | Makes an API request; not part of offline tests |

Original-DCGS A40 smokes257599/257600 passed output, runtime and memory checks.
Full launchers are prepared: one A40 each,48h,researchlong, all2037 dialogues,
separate `*_full_v2` outputs and declared record-and-continue for terminal
outputs. Original policy remains unchanged; see the linked review/recovery guide.
The user owns Slurm submissions; agents must not submit or mutate jobs.

The original-DCGS launchers now use fresh `*_smoke_v2` directories. V2 adds
verified critic-context diagnostics, typed failure records, a declared
record-and-continue option, and inspected fresh-directory recovery. See the
[policy and recovery guide](../docs/SAFEDIALBENCH_DCGS_PARITY.md). The
`safedial_dcgs_run_state.py` helper audits shared execution state; it does not
implement a separate DCGS policy.

## Why some older names remain

`run_safedial_dcgs.py` supplies history conversion and the model/critic loader
used by the current original-DCGS adapter. Its old standalone launcher is archived.
`run_safedial_smoothllm.py` and `run_safedial_tpo.py` supply shared runner
functions used by their current full/resumable entry points. Their validators
are also shared. Removing these files would break current imports or pinned
source hashes.

The `safedial_*.py` modules and `tpo_vendor/` contain runtime dependencies.
Keep them at their present paths while runs pin their hashes. Runtime code,
the active TPO/SmoothLLM launchers, and original-DCGS smoke manifests were not
changed by this cleanup.

## Offline checks

From the DCGS root:

```bash
module load Python/3.11.11-GCCcore-13.3.0
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest discover -s tests -v
```

The current suite has 130 tests. Historical custom-DCGS tests remain in the
archive's complete 143-test suite; active TPO retry coverage remains here.
