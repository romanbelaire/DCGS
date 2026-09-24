# SafeDialBench: runs, configuration, and saved results

This branch contains generation, validation, and judging tools for SafeDialBench: **2,037 dialogues / 10,029 turns**. The active DCGS runner implements main-method v3 with last-nonpadding-token high-level critic pooling and trained low-level token-critic reranking. Saved results now include finished 96-token VDCGS/RDCGS runs, finished VDCGS-384 generation, and partial ongoing experiments. See the [current progress](docs/SAFEDIALBENCH_PROGRESS.md), [score comparison and coverage](docs/SAFEDIALBENCH_SCORE_COMPARISON.md), and [25 September result snapshot](results/safedialbench/2026-09-25/README.md).

- [Current DCGS configuration](#current-dcgs-configuration)
- [Run VDCGS and RDCGS](#run-vdcgs-and-rdcgs)
- [Configure a run](#configure-a-run)
- [Outputs, validation, and resume](#outputs-validation-and-resume)
- [Saved benchmark snapshot](#saved-benchmark-snapshot)
- [Full method/checkpoint guide](docs/SAFEDIALBENCH_DCGS_MAIN.md) and [all active scripts](scripts/README.md)

## Current DCGS configuration

The entry point is [`scripts/run_safedial_dcgs_wildjailbreak.py`](scripts/run_safedial_dcgs_wildjailbreak.py). It loads [`safedial_dcgs.json`](safedial_dcgs.json), fills unspecified fields from `BaseConfig`, and applies the selected method/device and [`configs/safedial/dcgs_main.lock.json`](configs/safedial/dcgs_main.lock.json). The full effective configuration, artifact hashes, and source hashes are saved in each run's `run_config.json`.

| Setting | VDCGS | RDCGS |
| --- | --- | --- |
| CLI method | `--method vdcgs` | `--method rdcgs` |
| High-level candidate pool | Up to 5 nominal candidates | Up to 5 nominal + 5 adversarial candidates |
| High-level selection score | `Q` | `0.8 * Q - 0.2 * regret` |
| High-level selection | Softmax sampling; `epsilon=0`, `hl_greedy_q=false` | Same |
| `use_regret_critic` | Overridden to `false` | Overridden to `true` |
| Low-level selection | Trained token-critic argmax over 5 candidates; seeded random ties | Same |

| Shared setting | Current value / behavior |
| --- | --- |
| Actor | `HuggingFaceH4/zephyr-7b-beta`, revision `892b3d7a7b1cf10c7a701c60881cd93df615734c`; actual local snapshot is pinned by the lock |
| Precision / device | BF16 (`use_bf16=true`); generation CLI defaults to `cuda:0` |
| Policy | `environment_type=wildjailbreak`, hierarchical freeform policy; SafeDial supplies gold assistant history |
| Belief candidates | `n_candidates=5`, `freeform_n_instructions=5`; iterative candidate generation off |
| Belief generation | Temperature `0.8`, maximum `96` new tokens per list-generation request; these are inherited freeform defaults |
| High-level critics | Last-nonpad pooling; 1,500-token cutoff using the pinned tokenizer's left truncation |
| Regret settings | `regret_critic_beta=0.2`, `regret_min_target_mode=min_q_over_states`, `regret_zero_sum_targets=true` |
| Response generation | `ll_action_belief_only=true`; temperature `0.7`; numbered list of 5 responses with a shared `640`-token budget |
| Response reranking | `ll_candidate_rerank=true`, `n_ll_candidates=5`; trained `LLTokenCritic` shares the frozen actor backbone |
| Token-critic context | Maximum `8,192` tokens for observation + belief + response; **no truncation**. Observed overflows are recorded as terminal failures; see the progress report for coverage |
| Tokenizers | Actor/high-level tokenizer uses left padding; token critic uses a separate copy with right padding. Padding and truncation are separate settings |
| Batching | `q_value_chunk_size=4`, `batch_generation_chunk_size=1` |
| Critic heads | `critic_mlp_dims=null` permits inference from checkpoint shapes; `mlp_width_mult=1.0`; `critic_lora_r=0` |
| Training / seed | Generation only, no critic training despite training-related config fields; default base seed `0`, derived per turn |
| Other modes | Standard defender; GPT agents, baseline mode, greedy/oracle/random belief modes, raw-judge selection, and paper reward off |
| Protocol | `safedial_main_wildjailbreak_v3` using main revision `576ae184b8f49586456a136f8e14a93039cfeb88` |

The active artifacts are:

| Artifact | Repository path |
| --- | --- |
| High-level Q/regret checkpoint | `outputs/wildjailbreak_hierarchical_regret_critic/hierarchical_regret_critic/checkpoints/value_function_final.pt` |
| Low-level token critic | `models/models/wildjailbreak_ll_token_critic/ll_token_critic.pt` |
| Actor snapshot and all checksums | [`configs/safedial/dcgs_main.lock.json`](configs/safedial/dcgs_main.lock.json) |

The high-level checkpoint is the user-selected indexed final WildJailbreak checkpoint, with the same bytes as the older August 11 weights. Its training compatibility with main's changed pooling and regret target is **unverified**. Changing the inference code does not retrain it. The token critic is the artifact tracked in the merged main; its provenance notes remain in the lock.

## Run VDCGS and RDCGS

### Prerequisites

Run from the `DCGS` repository root on Linux/WSL. Use the project `.venv` with [`requirements.txt`](requirements.txt) installed and an appropriate CUDA PyTorch build for GPU generation. The cluster launchers expect the checkout at `/common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS` and load `Python/3.11.11-GCCcore-13.3.0`.

The execution host needs the pinned Zephyr snapshot, both real critic checkpoints (not just LFS pointers), and `benchmark-artifacts/SafeDialBench-Dataset/data/complete/datasets_en.jsonl`. The launchers also run judge input validation, which needs the benchmark's `FastChat/fastchat/llm_judge/data/judge_prompts.jsonl` under the same dataset checkout.

Materialize the critic weights before enabling offline mode:

```bash
git lfs pull --include="outputs/wildjailbreak_hierarchical_regret_critic/hierarchical_regret_critic/checkpoints/value_function_final.pt,models/models/wildjailbreak_ll_token_critic/ll_token_critic.pt"
```

For direct Python commands, use the same environment as the launchers:

```bash
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
```

The actor snapshot must already exist at the lock's `actor.path`; setting cache environment variables alone does not change that path. See the path override below for another machine.

### CPU preflight, without inference

These commands scan every dialogue, verify the pinned artifacts, and report context risks. They do not generate responses or call a judge API.

```bash
.venv/bin/python -B scripts/run_safedial_dcgs_wildjailbreak.py \
  --method vdcgs --device cpu --seed 0 --validate-only \
  --output-dir outputs/safedial_dcgs/zephyr_vdcgs_main_wildjailbreak_preflight_v3

.venv/bin/python -B scripts/run_safedial_dcgs_wildjailbreak.py \
  --method rdcgs --device cpu --seed 0 --validate-only \
  --output-dir outputs/safedial_dcgs/zephyr_rdcgs_main_wildjailbreak_preflight_v3
```

Inspect `preflight.json`, particularly `ll_critic_context.potential_overflow_turns` and `potential_overflow_turn_details`. A passed preflight means the scan completed, not that every generated candidate will fit. **Keep these CPU preflight directories separate from GPU run directories** because device and execution settings are pinned in the manifest. All real artifacts are still required for CPU preflight.

### GPU smoke and full runs through Slurm

Smoke runs select dialogue ID 1; full runs select all 2,037 dialogues. Run and inspect both new-method smokes before the full runs when GPU resources are available.

```bash
mkdir -p outputs/slurm
sbatch scripts/slurm/run_safedial_vdcgs_wildjailbreak_main_smoke.sbatch
sbatch scripts/slurm/run_safedial_rdcgs_wildjailbreak_main_smoke.sbatch
```

```bash
sbatch scripts/slurm/run_safedial_vdcgs_wildjailbreak_main_full.sbatch
sbatch scripts/slurm/run_safedial_rdcgs_wildjailbreak_main_full.sbatch
```

| Launcher | Allocation | Failure policy |
| --- | --- | --- |
| Smoke | 1 A100, 64 GB RAM, 8 CPUs, 4 hours, `researchshort` | Stop on the first failed turn |
| Full | 1 A40, 64 GB RAM, 8 CPUs, 48 hours, `researchlong` | Record terminal failures and continue to later turns/dialogues |

The launchers perform tokenizer preflight, generation, output/GPU-evidence validation, and a judge **dry-run**. They do not submit paid labelling requests. No v3 GPU smoke has run yet; actual memory and throughput remain unverified.

### Direct full generation without Slurm

These commands run in the foreground on a machine where a CUDA GPU is already available. Running without Slurm does not provide GPU access. Use the environment above, and run the methods separately if they would share one GPU.

```bash
.venv/bin/python -u -B scripts/run_safedial_dcgs_wildjailbreak.py \
  --method vdcgs --device cuda:0 --seed 0 \
  --on-turn-error record-and-continue \
  --output-dir outputs/safedial_dcgs/zephyr_vdcgs_main_wildjailbreak_full_v3
```

```bash
.venv/bin/python -u -B scripts/run_safedial_dcgs_wildjailbreak.py \
  --method rdcgs --device cuda:0 --seed 0 \
  --on-turn-error record-and-continue \
  --output-dir outputs/safedial_dcgs/zephyr_rdcgs_main_wildjailbreak_full_v3
```

For a direct smoke, add `--ids 1`, use `--on-turn-error stop`, and change the output suffix to `_smoke_v3`. A direct generation command does not automatically run preflight, post-run validation, or judging; use the separate commands in this README.

## Configure a run

There is **no `--config` argument** on this runner. Configuration sources have these roles:

| What to change | How |
| --- | --- |
| VDCGS versus RDCGS | `--method`; this overrides the JSON's `use_regret_critic` |
| Device, seed, output location | `--device`, `--seed`, `--output-dir` |
| Dataset | `--dataset /path/to/datasets_en.jsonl`; default is the benchmark path above |
| Dialogue selection | One of `--ids 1,2`, `--limit 10`, or `--per-task 1`; omit all three for the full dataset |
| Continue after terminal failures | `--on-turn-error record-and-continue`; bare CLI default is `stop` |
| Artifact locations | `--artifact-lock /path/to/lock.json`; Slurm launchers also accept `DCGS_ARTIFACT_LOCK` |
| Model/policy parameters | Edit `safedial_dcgs.json` before starting a fresh run; unspecified fields come from `src/configs/base_config.py` |
| Token budgets / context limits | Belief budget comes from `freeform_max_new_tokens`; the LL pool budget and critic context limits are currently defined in code, not runner CLI options |

For another machine, copy the artifact lock and change only `actor.path`, `high_level.path`, and/or `token_critic.path` to the materialized artifacts. Preserve the hashes to keep the same experiment. Paths in the lock override the checkpoint paths in `safedial_dcgs.json`, and its actor path controls the loaded model. Relative artifact paths resolve against the repository root.

```bash
cp configs/safedial/dcgs_main.lock.json configs/safedial/dcgs_main.local.lock.json
# Edit the paths in dcgs_main.local.lock.json, retaining the pinned hashes.
export DCGS_ARTIFACT_LOCK="$PWD/configs/safedial/dcgs_main.local.lock.json"
sbatch scripts/slurm/run_safedial_vdcgs_wildjailbreak_main_smoke.sbatch
```

For direct Python commands, pass `--artifact-lock "$DCGS_ARTIFACT_LOCK"` explicitly; the environment variable is read by the shell launchers, not by the Python CLI.

For a planned parameter experiment, settings such as `regret_critic_beta` or `freeform_max_new_tokens` belong in `safedial_dcgs.json`; start in a new output directory and rerun preflight. Keep `n_candidates` and `freeform_n_instructions` equal. Changing a score/budget is a different experiment even if the runner accepts it. Fixed-method guards reject incompatible settings, including disabling trained LL reranking, changing its five-candidate count, enabling LL history prompts, or switching to GPT agents, wrapped defenders, baseline mode, or unsupported belief-selection modes. Supporting those variants requires a code/parity review, not just a JSON toggle.

Never change the config, code, dataset, seed, selection, method, device, or artifact lock for an existing output directory: manifest checks will reject continuation. Use a fresh directory for any new configuration. The full Slurm files accept an optional output directory as their first argument; their remaining run flags are fixed in the script. Smoke files use their fixed smoke directories. Both include a cluster-specific `cd` path and resource directives that must match the execution host.

## Outputs, validation, and resume

| Location / file | Purpose |
| --- | --- |
| `outputs/safedial_dcgs/zephyr_{vdcgs,rdcgs}_main_wildjailbreak_{smoke,full}_v3/` | Fresh v3 run directories |
| `run_config.json`, `preflight.json` | Pinned settings/artifacts/sources and optional tokenizer preflight report |
| `turns.jsonl`, `answers.jsonl` | Successful turns and native answer exports; only entirely successful dialogues enter `answers.jsonl` |
| `failures.jsonl`, `failure.json` | Failure ledger and, when present, a failure blocking ordinary resume |
| `events.jsonl`, `attempts.jsonl`, `model_loading.jsonl`, `runtime.jsonl` | Generation/scoring audit, attempts, loading, and runtime evidence |
| `outputs/slurm/` | Scheduler stdout/stderr; creating this directory does not change the result location |
| `results/safedialbench/` | Separately captured, committed snapshots, not the live generation destination |

Belief generation retains **three total attempts** when every candidate is `[SKIP]`. If fewer than five LL responses parse, main generates one 128-token fallback response and repeats it to fill the pool. There are no extra runner-level retries after those attempts/fallbacks. A remaining blank candidate/response, exhausted beliefs, or LL context overflow becomes a terminal failed turn. Full runs continue with later gold-history turns and dialogues; smoke runs stop. CUDA/OOM and integrity errors stop both.

Rerun an identical command to resume an ordinary interrupted run. Successful and terminal-failed turns are retained and are not regenerated. If a blocking `failure.json` is present, inspect it and use `--recover-from` with `--recovery-reason` to prepare a fresh directory under the same method/settings; recovery itself does not generate responses. The [method guide](docs/SAFEDIALBENCH_DCGS_MAIN.md) explains the audit model. Old v2 directories cannot be resumed or reinterpreted as v3.

Validate each GPU run afterward (substitute `rdcgs` for the RDCGS directory):

```bash
.venv/bin/python -B scripts/validate_safedial_dcgs_wildjailbreak.py \
  --output-dir outputs/safedial_dcgs/zephyr_vdcgs_main_wildjailbreak_full_v3 \
  --require-gpu
```

Use `--allow-incomplete` when auditing partial/failed output; it does not mark incomplete coverage successful. A generation run that processes everything but has failed turns returns exit code **2**. Read `execution_finished`, `complete_without_failures`, and failure counts separately.

Input validation for later API labelling is CPU-only:

```bash
.venv/bin/python -B scripts/judge_safedial.py \
  --answers outputs/safedial_dcgs/zephyr_vdcgs_main_wildjailbreak_full_v3/answers.jsonl \
  --dry-run
```

Paid judging is separate from DCGS generation; see the [labelling guide](docs/SAFEDIALBENCH_FULL_JUDGING.md). SmoothLLM and TPO remain separate supported baseline runners, listed in the [script index](scripts/README.md).

SmoothLLM labelling can run locally on Windows/Git Bash or Linux without Slurm or a GPU. From this directory, prepare and check the saved September 19 answers:

```bash
python scripts/run_safedial_smoothllm_judging.py --snapshot results/safedialbench/2026-09-19 --fetch-benchmark --dry-run
```

Then remove `--dry-run` to start paid API judging using `OPENAI_API_KEY` from `.env` or the environment. This labels 2,036 dialogues / 10,024 turns, excluding failed dialogue 344. New outputs go under `outputs/safedial_judging/smoothllm_zephyr_full_turn_resume_snapshot_2026-09-19/`; the committed snapshot is unchanged. See the guide for dependencies, resume, Linux full-journal mode, and the snapshot audit's limits.


## Saved benchmark snapshot

The latest [25 September snapshot](results/safedialbench/2026-09-25/README.md) contains generation outputs, native judging, LlamaGuard, assistance evaluation and goal extraction. The [partial-judging supplement](results/safedialbench/2026-09-25-partial-judging/README.md) adds the four completed historical VDCGS/RDCGS judging subsets. The main snapshot includes partial active runs; consult its timestamps and file inventory. The [score comparison](docs/SAFEDIALBENCH_SCORE_COMPARISON.md) distinguishes native scores, LlamaGuard DSR and combined DSR, including missing responses and unresolved judge errors.

The following table is historical, from the September 19 snapshot.

Status checked **19 September 2026, 01:32 SGT (UTC+8)**. Result files captured independently during `2026-09-19T01:30:35+08:00`–`2026-09-19T01:31:33+08:00`. At that capture, both old full DCGS jobs reported running. Their present scheduler state has not been checked.

This branch contains SafeDialBench generation, validation, and judging code plus saved results. A full benchmark has **2,037 dialogues and 10,029 turns**. Active-run counts are snapshots, not final results.

SmoothLLM judging was subsequently completed locally and verified on **19 September 2026, 22:07 SGT**: **10,024/10,024 successful judgments**, covering **2,036 dialogues**, with overall score **3.5378**. Dialogue 344 remains excluded. See the [completed judging archive](results/safedialbench/2026-09-19-smoothllm-judging/README.md); the earlier generation snapshot remains unchanged. Other rows below retain their original capture status.

### Saved benchmark runs

| Benchmark / method | Status | Successful generation turns | Validation / judging / remaining issue |
| --- | --- | ---: | --- |
| Zephyr baseline | Generation complete; judging incomplete | 10,029/10,029 | 10,028/10,029 turn judgments; one unresolved judge response. |
| GPT-4o baseline | Processing complete; filtered case excluded | 10,028/10,029 | One filtered turn in dialogue 1436. Judging complete for 2,036 exported dialogues / 10,024 turns, not the full benchmark. |
| CAT + Zephyr | Generation and judging complete | 10,029/10,029 | Validation passed; all 10,029 turn judgments complete. |
| SmoothLLM + Zephyr | Successful-dialogue judging complete | 10,028/10,029 | 10,024/10,024 judgments over 2,036 dialogues; overall 3.5378. Dialogue 344 excluded for generation failure; full candidate audit unavailable in the snapshot. |
| TPO v8 + Zephyr | Stopped on parser failure (257501) | 281/10,029 | Dialogue 57, turn index 1: missing IMPROVED_VARIABLE opening tag. Exit 2; no final judge score. |
| VDCGS, original WildJailbreak | Running (257632) | 6,237/10,029 | Five terminal all-SKIP belief failures, retained in the failure ledger. Generation continues; judging pending. |
| RDCGS, original WildJailbreak | Running (257633) | 2,982/10,029 | No terminal failures recorded at this check; judging pending. |
| TPO upstream-handling smoke | Passed on L40 (257858) | 5/5 | 5/5 turns; 75 candidates; GPU audit and judge dry-run passed. No skipped candidates in this real smoke; skip handling has offline tests. Full run not launched. |

Counts include inherited results for resumed runs and deduplicate repeated records by dialogue/turn. A saved turn or completed Slurm job does not by itself establish successful validation or a final benchmark score. Turn indices in failure notes are zero-based.

### Smoke and historical runs

The directories below retain prior smoke results, failed attempts, or superseded policies. They are not additional completed full-benchmark results and must not be pooled with the current methods.

| Saved run directory | Saved successful turns | Recorded status |
| --- | ---: | --- |
| `cat_zephyr_smoke` | 5/5 | Historical/smoke validation passed |
| `smoothllm_zephyr_full` | 7,841/10,029 | Historical failure; 1 failed turn retained |
| `smoothllm_zephyr_smoke` | 5/5 | Historical/smoke validation passed |
| `tpo_zephyr_full` | 9/10,029 | Historical failure; 1 failed turn retained |
| `tpo_zephyr_full_v4` | 125/10,029 | Historical failure; 1 failed turn retained |
| `tpo_zephyr_full_v5` | 126/10,029 | Historical failure; 1 failed turn retained |
| `tpo_zephyr_full_v6` | 130/10,029 | Historical failure; 1 failed turn retained |
| `tpo_zephyr_full_v7` | 140/10,029 | Historical failure; 1 failed turn retained |
| `tpo_zephyr_smoke` | 2/5 | Historical failure; 1 failed turn retained |
| `tpo_zephyr_smoke_failed_249773` | 0/5 | Historical CUDA startup failure; no generated turns |
| `tpo_zephyr_smoke_v2` | 4/5 | Historical failure; 1 failed turn retained |
| `tpo_zephyr_smoke_v3` | 5/5 | Historical/smoke validation passed |
| `zephyr_7b_beta_smoke` | 62/62 | Smoke judging complete |
| `zephyr_7b_beta_wildjailbreak_critic_smoke` | 5/5 | Historical outputs; no final validation report |
| `zephyr_rdcgs_aug11_a5000_full` | 100/10,029 | Historical failure; 1 failed turn retained |
| `zephyr_rdcgs_aug11_a5000_full_retry_v1` | 100/10,029 | Superseded prepared continuation; inherited outputs |
| `zephyr_rdcgs_original_wildjailbreak_smoke` | 0/5 | Prepared/superseded; no generated turns |
| `zephyr_rdcgs_original_wildjailbreak_smoke_v2` | 5/5 | Historical/smoke validation passed |
| `zephyr_vdcgs_aug11_a5000_full` | 29/10,029 | Historical failure; 1 failed turn retained |
| `zephyr_vdcgs_aug11_a5000_full_retry_v1` | 29/10,029 | Superseded prepared continuation; inherited outputs |
| `zephyr_vdcgs_original_wildjailbreak_smoke` | 0/5 | Prepared/superseded; no generated turns |
| `zephyr_vdcgs_original_wildjailbreak_smoke_v2` | 5/5 | Historical/smoke validation passed |
| `zephyr_wildjailbreak_two_stage_smoke` | 5/5 | Historical/smoke validation passed |

Earlier A40 TPO smoke submission **257857** failed because L40 job **257858** already held the output lock. It is not a separate model result. Original VDCGS/RDCGS v2 smoke jobs **257599/257600** passed, 5/5 turns each; their original 1,500-token critic truncation remains part of the preserved policy.

## Results and reproducibility

- [Latest snapshot evidence](results/safedialbench/2026-09-25/snapshot.json): generation, native judging, LlamaGuard, assistance and goal artifacts with checksums. [Scheduler state](results/safedialbench/2026-09-25/scheduler.txt) was captured separately.
- [September 19 saved results](results/safedialbench/2026-09-19/README.md): historical saved results from 31 run directories. Independent file snapshots are not resumable checkpoints.
- [Previous status evidence](docs/verification/benchmark_status_20260918.json): status recorded on 18 September.
- [Saved result snapshot](results/safedialbench/2026-09-18/README.md): answers, judgments, manifests, and compact turn records from 31 run directories. **This snapshot predates the status update above**; its live-run outputs may be less complete. See its [timestamp and checksums](results/safedialbench/2026-09-18/snapshot.json).
- [Script index](scripts/README.md): generation, validation, and Slurm entrypoints.
- [Upstream TPO handling](docs/SAFEDIALBENCH_TPO_UPSTREAM_HANDLING.md): skip-and-record policy and resume behavior.
- [Original DCGS policy](docs/SAFEDIALBENCH_DCGS_PARITY.md): original-code reuse, failure handling, and critic-context limitations.

The current offline suite passed **321 tests** on September 25, including snapshot integrity, goal extraction, normalized assistance judging and isolated replay checks. This does not constitute new GPU validation. Large saved JSONL files are gzip-compressed. Full model-call journals, checkpoints, caches, and the external dataset remain outside the committed result snapshot.

Root `handover.md`, `runbook.md`, `AGENTS.md`, `agents.md`, and `.agents.md` are local operational files and ignored by Git. Immutable archived audit copies remain part of the historical snapshot.
