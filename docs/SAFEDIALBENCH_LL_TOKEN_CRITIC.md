# Downloaded LL token critic

Fetched from romanbelaire/DCGS commit
`576ae184b8f49586456a136f8e14a93039cfeb88` on 2026-09-17.
The upstream tree uses `models/models/`; local paths remove the repeated folder:

- `models/wildjailbreak_ll_token_critic/ll_token_critic.pt`
- `models/cares_ll_token_critic/ll_token_critic.pt`
- Matching, unchanged upstream loader: `src/value/ll_token_critic.py`.

Both checkpoints are **byte-identical**: 134,287,165 bytes, SHA-256
`a0a4cc34e4a096122adeb21d5cff2acfe6cf760c22f70b9a76733555b9e1313f`.
This matches both upstream LFS pointers. Metadata records hidden_size4096,
mlp_width_mult2.0, objective shapley, n_examples804, epochs20, seed42,
gamma0.99 and k5. It does not identify the training dataset. Do not claim these
are independently trained CARES and WildJailbreak weights without provenance.
See `models/ll_token_critic_provenance.json` for exact paths, metadata and hashes.

## Verification

CPU `torch.load(weights_only=True)` passed. Both harm/follow heads load strictly
into the upstream architecture; each has16,785,409 parameters, all finite.
Scoring synthetic zero residuals produces a finite scalar. These checks do not
establish GPU execution or model quality; no Zephyr backbone was loaded.
The following reproducible inspection requires the existing project environment:

```bash
LD_LIBRARY_PATH=/opt/apps/software/Python/3.11.11-GCCcore-13.3.0/lib .venv/bin/python -B docs/verification/ll_token_critic_20260917/inspect_downloaded_ll_critic.py
```

## DCGS integration prepared

The user explicitly selected the WildJailbreak token path. Its absent dataset
metadata remains recorded provenance, not an approval blocker.

The separate two-stage runner now loads the trained harm/follow heads alongside
the existing Q/regret critic and uses contextual token scoring for response
selection. The original reduced runner is unchanged. See
[the two-stage experiment guide](SAFEDIALBENCH_DCGS_TWO_STAGE.md) for the exact
policy, precision/context adaptations, verification results and GPU smoke command.
Actual GPU smoke remains pending; a full batch has not been prepared.
