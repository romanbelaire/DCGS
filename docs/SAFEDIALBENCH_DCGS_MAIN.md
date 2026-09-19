# SafeDial DCGS: main policy with trained low-level reranking

The active adapter uses main revision `576ae184b8f49586456a136f8e14a93039cfeb88`, with last-nonpadding-token high-level critic pooling and the trained WildJailbreak low-level token critic. Both VDCGS and RDCGS use protocol `safedial_main_wildjailbreak_v3` and separate `*_main_wildjailbreak_*_v3` directories. Saved v2 results remain historical results of their original method.

For the current settings table, configuration precedence, and Slurm/direct CLI examples, see [the repository README](../README.md#current-dcgs-configuration).

## Method and checkpoint choices

- VDCGS uses the indexed Q head without adversarial pool expansion. RDCGS keeps the nominal/adversarial pools and samples using `0.8 * Q - 0.2 * regret`. High-level greedy/static/oracle modes remain off.
- High-level scoring uses main's last-nonpad pooling and its unchanged 1,500-token context cutoff. The configuration uses `regret_min_target_mode=min_q_over_states`; generation does not retrain the critic.
- Low-level generation follows main's numbered-list implementation: five candidates, a 640-token pool budget, and the upstream single-response fallback if parsing yields too few candidates. The trained `LLTokenCritic.score_actions` scores the pool; main selects an argmax, breaking ties with the per-turn seeded RNG. Repeated strings are merged as in main. The checkpoint's Shapley/TD/span objective determines token aggregation.
- Both critics share the pinned Zephyr BF16 backbone. The LL critic receives a separate copy of the pinned actor tokenizer with right padding. It does not fetch an unpinned tokenizer or change the actor/HL tokenizer's left padding.
- Gold assistant history, reference-answer isolation, seed/replay behavior, and record-and-continue semantics are preserved. Empty candidates remaining after the upstream fallback become terminal `blank_ll_candidate` failures. Intact LL critic inputs above 8,192 tokens become terminal `ll_context_overflow` failures; inputs are not truncated and no extra attempts are introduced.

The user selected the WildJailbreak **final** high-level checkpoint from `models/MANIFEST.json`. Its LFS pointer is restored at the indexed path after main removed it. Its SHA-256 is `b1ecbe5461dfefe0edf9a2f6fb83277cf494d9744deea0cf56bb58d4bbf6d75d`, identical to the earlier August 11 WildJailbreak weights. The index has no pooling/training-target metadata: this is **new-method inference with the existing indexed weights**, not evidence of retraining under main's revised objective.

The LL checkpoint is main's `models/models/wildjailbreak_ll_token_critic/ll_token_critic.pt`, SHA-256 `a0a4cc34e4a096122adeb21d5cff2acfe6cf760c22f70b9a76733555b9e1313f`. Existing provenance notes about its training dataset remain in the lock.

## Prepare the execution checkout

Materialize the two LFS artifacts on the execution host:

```bash
git lfs pull --include="outputs/wildjailbreak_hierarchical_regret_critic/hierarchical_regret_critic/checkpoints/value_function_final.pt,models/models/wildjailbreak_ll_token_critic/ll_token_critic.pt"
```

`configs/safedial/dcgs_main.lock.json` pins the actor snapshot and both critics. Critic paths are repository-relative; the actor keeps the existing execution-host cache path. If the artifacts live elsewhere, copy the lock, change only their paths, and pass `--artifact-lock /path/to/lock.json`. Slurm launchers accept the same override through `DCGS_ARTIFACT_LOCK`. Hashes are checked before generation; pointer-only or different weights fail explicitly.

CPU/tokenizer preflight for all dialogues (no inference; requires the real local artifacts):

```bash
.venv/bin/python -B scripts/run_safedial_dcgs_wildjailbreak.py \
  --method vdcgs --device cpu --validate-only \
  --output-dir outputs/safedial_dcgs/zephyr_vdcgs_main_wildjailbreak_preflight_v3
```

Inspect `preflight.json` before GPU work, especially `ll_critic_context.potential_overflow_turns` and its turn details. The estimate reserves the full belief budget and the full 640-token response-pool budget for a candidate because the parser does not enforce equal candidate lengths. Decoded-text tokenization can vary, so runtime checks remain authoritative. `passed: true` means the preflight scan completed; it does **not** guarantee every turn will fit. Repeat with `--method rdcgs` and a separate output directory.

Use the new launchers for actual GPU work. Submissions remain user-owned:

```bash
mkdir -p outputs/slurm
sbatch scripts/slurm/run_safedial_vdcgs_wildjailbreak_main_smoke.sbatch
sbatch scripts/slurm/run_safedial_rdcgs_wildjailbreak_main_smoke.sbatch
# After checking the real smoke's memory, replay, and output validation:
sbatch scripts/slurm/run_safedial_vdcgs_wildjailbreak_main_full.sbatch
sbatch scripts/slurm/run_safedial_rdcgs_wildjailbreak_main_full.sbatch
```

`outputs/slurm` holds scheduler logs. Run records and generated answers go to `outputs/safedial_dcgs/`; `results/safedialbench/` holds separately captured, committed snapshots.

The smoke launchers retain their one-A100 allocation; the full launchers retain one A40 / 64 GB host RAM / 48 hours. Actual capacity and throughput for LL reranking require the new GPU smoke. The launchers validate successful GPU evidence and perform only a judge dry-run, with no paid judging.

## Audit and continuation

The manifest pins the method specification, source hashes, both critic hashes, and effective configuration. Model-loading records include LL objective and inferred HL head dimensions. Each LL scoring event saves the candidate list, scores, objective, and action-token/context hashes. The validator replays the exact upstream selection and rejects missing/tampered scoring metadata.

```bash
.venv/bin/python -B scripts/validate_safedial_dcgs_wildjailbreak.py \
  --output-dir outputs/safedial_dcgs/zephyr_vdcgs_main_wildjailbreak_full_v3 \
  --require-gpu
```

Full launchers use `--on-turn-error record-and-continue`: after a terminal failure they process later turns and dialogues using gold history. An overflow skips the whole turn, without silently dropping candidates. Smoke launchers use `stop` to expose the first failure. Infrastructure errors (including CUDA OOM) and integrity errors still stop both. Completed/terminal turns are never retried on resume.

An execution that finishes with failed turns returns exit code 2. The validator reports `execution_finished: true`, `complete_without_failures: false`, and `ll_critic_overflow_turns` / `ll_critic_overflow_max_input_tokens`; it does not label partial coverage successful. `answers.jsonl` exports only dialogues whose turns all succeeded. Overflow replay re-tokenizes the original request and checks the recorded token counts and limit. Failure IDs are stable across JSON key ordering.

Same-method continuation and inspected fresh-directory recovery remain available through the existing runner. A v2 run cannot be resumed, imported, or validated as v3. Use checkout `4e2f493` to audit the old method. Old `*_original_*` Slurm files and archived documentation describe v2; use the `*_main_*` launchers above for v3. The internal `dcgs_original` journal field is retained for the shared audit schema; the manifest protocol and method specification identify v3.

The active runner rejects the optional main routes flagged in review: GPT agents, wrapped defenders, legacy LL reranking without a trained checkpoint, greedy/random/oracle HL selection, baseline mode, and paper reward. The separate main-path bugs remain deferred; this readiness claim applies to the VDCGS/RDCGS launchers documented here.

Offline verification uses `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -B -m unittest discover -s tests -v` on Linux. Tests use injected responses, small tensors, and a tiny randomly initialized CPU Mistral with temporary critic checkpoints. The real backend test exercises both checkpoint loaders, shared-backbone scoring, separate padding sides, and strict frozen heads; it does not validate the indexed trained artifacts or GPU capacity.

Verification: all 146 tests passed on Linux with CPU PyTorch; `bash -n` passed for all four new launchers. Only the tiny synthetic model has run; no inference with the indexed trained artifacts or GPU smoke has run. GPU memory, throughput, and generated-result validation remain pending until a GPU is available.
