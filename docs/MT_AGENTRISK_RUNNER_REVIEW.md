# AgentRisk runner review — 2026-09-21

The runner is ready for another **single-task diagnostic after installing this
client update and starting a fresh server**. This is an offline code/fixture
review, not a successful new Docker/GPU benchmark run. Task mistakes remain
measured outcomes; the adapter does not repair or resample selected actions.

## Confirmed fixes

| Area | Finding | Correction |
|---|---|---|
| DCGS replay | Belief retries mutated recorded text during replay, causing a false audit mismatch. | Replay a private event copy and retain the full audit/selection comparison; regression covers VDCGS and RDCGS. |
| Run status | Upstream can return a controller in ERROR without raising; the launcher returned exit code 0. | Write execution_result.json and return nonzero for controller errors, incomplete runs, or missing artifacts. Exceptions are recorded and re-raised. |
| Startup | Health checked neither the bearer token nor whether the server loaded the replay fix. | Authenticated model-list preflight before Docker startup; require current server/replay versions and available request budget. |
| Profile credentials | Switching from local inference back to the official baseline could reuse the cluster endpoint/token. | Official overrides use MT_OFFICIAL_AGENT_MODEL, MT_OFFICIAL_AGENT_BASE_URL and MT_OFFICIAL_AGENT_API_KEY; OPENROUTER_API_KEY remains the default. Local methods retain MT_AGENT_BASE_URL and MT_AGENT_API_KEY. |
| JSON boundary | Standard JSON decoding silently accepted duplicate object keys and nonfinite numbers. | Reject ambiguous/nonstandard JSON at request and selected-action boundaries. No candidate filtering or repair. |
| Failure evidence | Replay-validation errors lost the completed selection audit and were reported like model-format failures. | Preserve failed_audit.json and distinguish runner errors (500) from model-protocol/terminal-output failures (422). Both are cached without resampling. |

## Verification

- 15 policy/protocol/server tests pass, including VDCGS/RDCGS retry replay,
  immutable evidence, tamper rejection, JSON arrays, full tool-result history,
  SmoothLLM/TPO fixture replay, and audit preservation.
- 12 launcher tests pass, including profile isolation, authentication, stale
  server/budget rejection, error exit status, missing artifacts, and exceptions.
- Loopback HTTP test uses pinned LiteLLM 1.74.3/OpenAI 1.99.9 and the actual
  launcher preflight, with a fake backend: authentication, duplicate-JSON
  rejection, tool-result round trip and cached retries; no real inference.
- All five saved job260411 requests replay offline with unchanged backend
  requests and preserved events; all four previously saved selected actions match.
  The formerly blocked native task now passes replay, without executing its action.
- Slurm launcher passes bash -n. 95 frozen policy/prompt/SafeDial source hashes
  match the previous job manifest. No model weights, token budgets, critic
  selection policy, or official evaluator source were changed.

## Evaluation limitations retained

- Official final scoring remains a separate agentrisk/post_eval.py step.
  Controller completion is not a benchmark outcome; execution_result.json
  explicitly records evaluation_complete=false and benchmark_outcome=null.
- The upstream turn manager uses keyword matches over recent events, including
  instructions. These can advance a turn without proving successful execution.
  Its per-turn limit reads legacy state.iteration, which was None in the uploaded
  states; the configured overall iteration limit remains the bound. These
  upstream behaviors are preserved for comparison with the official baseline.
- post_eval.py iterates every child of its task root and labels missing logs as
  FAILED. A single-task judging command must restrict the task root accordingly;
  do not include unrun tasks in the reported denominator.
- The actor remains limited by its token budgets and the DCGS critics' context
  limits. Model-format errors, bad tool choices and context-limit failures stay
  visible; they are not relabelled as safety refusals.
- New Windows/Docker execution, sustained GPU inference and other benchmark
  categories still require live verification. The real-task diagnostic remains
  limited to multi.71; broader method/category performance is unmeasured.

## Install the reviewed client

From Ubuntu/WSL with the VPN connected:

```bash
cd ~/benchmarks/mt-agentrisk/ToolShield
scp darrius.ng.2024@10.193.104.102:/common/home/users/d/darrius.ng.2024/projects/RL-Defense/mt_agentrisk_transfer/mt_agentrisk_integration_update.tar.gz /tmp/
scp darrius.ng.2024@10.193.104.102:/common/home/users/d/darrius.ng.2024/projects/RL-Defense/mt_agentrisk_transfer/mt_agentrisk_integration_update.tar.gz.sha256 /tmp/
(cd /tmp && sha256sum -c mt_agentrisk_integration_update.tar.gz.sha256) && \
tar -xzf /tmp/mt_agentrisk_integration_update.tar.gz -C ~/benchmarks/mt-agentrisk/ToolShield
.venv/bin/python local_setup/test_configuration.py
```

Expected: 12 tests pass. This updates local_setup only; no dependency reinstall
or Docker rebuild is needed. Do not reuse the old server or old-job smoke report.

After this update, submit a fresh server from the DCGS directory on origami:

```bash
sbatch --gres=gpu:a100:1 scripts/slurm/run_mt_agentrisk_server.sbatch vdcgs
```

Once ready, reconnect using its new connection.json/token, save one job-matched
probe, and use --diagnostic-task-smoke for multi.71 even if that probe fails.
Use a fresh task output directory. Inspect execution_result.json and preserve
state/trajectory/server audits before final judging. The user owns Slurm actions.
