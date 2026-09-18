# Script cleanup archive — 2026-09-18

This archive preserves the files removed from the active script directory and
their cleanup-time dependencies. It is an audit snapshot, not a recommended
set of new experiment launchers.

- `snapshot/`: 201 exact source, test, config, instruction and guide files,
  retaining the original repository layout. Includes all pre-cleanup scripts,
  original `src`, tests, prompt templates, artifact locks and vendor license.
- `MANIFEST.json`: SHA-256 for each snapshot file, retained/archived script
  inventory, the 108 protected source hashes, and historical run references.
- `run_manifests/`: 26 byte-preserved top-level experiment configurations.
- `cleanup_actions.json`: exact removed paths and disposable bytecode removed.
- `verification/`: test logs and post-cleanup integrity checks.
- `verify_cleanup.py`: read-only snapshot, pinned-source, import and launcher
  checks; run `python3 -B archive/scripts_cleanup_20260918/verify_cleanup.py`
  from the DCGS root. It intentionally reports a mismatch if protected files
  are legitimately changed by a later experiment.

## Removed from the active directory

| Group | Archived files |
| --- | --- |
| Custom DCGS methods | `run_safedial_dcgs_methods.py`, `safedial_dcgs_methods.py`, `safedial_dcgs_methods_runtime.py`, `safedial_dcgs_native_context.py`, `validate_safedial_dcgs_methods.py` |
| Earlier two-stage DCGS | `run_safedial_dcgs_two_stage.py`, `safedial_dcgs_two_stage.py`, `safedial_dcgs_two_stage_runtime.py`, `validate_safedial_dcgs_two_stage.py` |
| Completed migrations | `prepare_safedial_tpo_v4.py` through `v7.py`, `prepare_safedial_empty_retry.py` |
| Obsolete launchers | Legacy DCGS smoke, two-stage smoke, six custom VDCGS/RDCGS full/retry launchers, original SmoothLLM full launcher, TPO v7 full and v3 smoke launchers |
| Historical test modules | `test_safedial_dcgs_methods.py`, `test_safedial_dcgs_two_stage.py`; the original mixed retry tests are also preserved in the snapshot |

All source was copied and hash-verified before removal from the active tree.
Only generated `.pyc` cache files were deleted without archival. Experiment
answers, event journals, checkpoints, datasets and existing provenance remain
at their original locations. The historical two-stage validation report was
regenerated successfully from its existing outputs; no model inference ran.

## Versions and provenance

The snapshot records the code present at cleanup time. It is not automatically
the source version of every older run. `historical_source_references` records
each old manifest's expected digest and whether this snapshot matches it.
Use the exact earlier source when a digest differs. Existing source snapshots
and nested provenance are preserved under:

- `docs/verification/empty_retry_20260918/{original_source,prepared_source}/`
- `docs/verification/dcgs_two_stage_20260917/`
- `docs/verification/dcgs_methods_20260917/`
- `docs/verification/dcgs_a5000_20260917/`
- `outputs/safedial_baseline/*/provenance/` and `outputs/safedial_dcgs/*/provenance/`

Do not copy archived modules over files used by a running job. Archived
launchers retain their original absolute `cd` and output paths for audit;
**do not submit them from this directory**. Use a separate review checkout
with the original layout and exact manifest-matched source for reproduction.
External weights/datasets are referenced, not duplicated into this archive.

## Read-only audit and tests

From the live DCGS root, with its Python module and virtual environment:

```bash
module load Python/3.11.11-GCCcore-13.3.0
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m unittest discover \
  -s archive/scripts_cleanup_20260918/snapshot/tests -v
```

This runs the preserved 143-test suite without a benchmark model or API call.
To audit the historical two-stage smoke without rewriting its report, import
the archived validator and call its read-only function:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'archive/scripts_cleanup_20260918/snapshot/scripts')
from validate_safedial_dcgs_two_stage import validate_dcgs
print(validate_dcgs(Path('outputs/safedial_dcgs/zephyr_wildjailbreak_two_stage_smoke'), True))
PY
```

For current experiment entry points, use [scripts/README.md](../../scripts/README.md).
