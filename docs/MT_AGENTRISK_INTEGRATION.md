# MT-AgentRisk remote inference

Current setup: read [the 2026-09-21 runner review](MT_AGENTRISK_RUNNER_REVIEW.md).
It requires an updated local launcher and a fresh server. Job-specific commands
below describe earlier diagnostics and must not be reused with old job IDs.

The benchmark and its Docker tools run in your existing Ubuntu/WSL checkout.
Local model inference runs on one cluster L40S. GPT-4o uses OpenRouter directly.
This update adds method selection; it does not require rebuilding the Docker
image or reinstalling the already working benchmark environment.

## Install the update

In Windows PowerShell, while connected to the institutional VPN:

```powershell
scp darrius.ng.2024@10.193.104.102:/common/home/users/d/darrius.ng.2024/projects/RL-Defense/mt_agentrisk_transfer/mt_agentrisk_integration_update.tar.gz "$HOME\Downloads\"
scp darrius.ng.2024@10.193.104.102:/common/home/users/d/darrius.ng.2024/projects/RL-Defense/mt_agentrisk_transfer/mt_agentrisk_integration_update.tar.gz.sha256 "$HOME\Downloads\"
```

In Ubuntu:

```bash
cd /mnt/c/Users/darri/Downloads
sha256sum -c mt_agentrisk_integration_update.tar.gz.sha256
tar -xzf mt_agentrisk_integration_update.tar.gz -C ~/benchmarks/mt-agentrisk/ToolShield
cd ~/benchmarks/mt-agentrisk/ToolShield
.venv/bin/python local_setup/test_configuration.py
.venv/bin/python local_setup/run_baseline.py --method vdcgs --task-id multi.71
```

The archive updates only files under `local_setup`. Existing outputs, task
workspaces, `.venv`, and the cached runtime image are preserved. The last command
prints a plan only. Eight configuration tests should pass.

## Start the VDCGS inference job

On origami, submit manually:

```bash
cd /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS
sbatch scripts/slurm/run_mt_agentrisk_server.sbatch vdcgs
```

Wait for `MT_AGENTRISK_READY` in `outputs/slurm/mt-agentrisk-server-JOBID.out`.
Loading and hash validation can take several minutes. The service allows 40
unique requests and runs for two hours after loading. The Slurm allocation is
three hours including loading. It writes its node/port to
`outputs/mt_agentrisk_server/JOBID/connection.json` and a private bearer token to
`api_token` alongside it. Never paste that token into chat.

## Connect from Ubuntu

Use a tunnel launched **inside Ubuntu**, where the benchmark runs. The previously
successful Windows PowerShell tunnel only proved Windows connectivity. WSL must
also be able to reach the cluster over the VPN. Keep Docker Desktop running with
WSL integration and host networking enabled as in the successful baseline run.

In a separate Ubuntu terminal, enter the actual numeric job ID when prompted:

```bash
read -rp 'Slurm job ID: ' MT_JOB_ID
[[ "$MT_JOB_ID" =~ ^[0-9]+$ ]] || { echo 'Expected a numeric job ID'; exit 1; }
MT_CLUSTER=darrius.ng.2024@10.193.104.102
MT_SERVER_DIR=/common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS/outputs/mt_agentrisk_server/$MT_JOB_ID
ssh "$MT_CLUSTER" "cat '$MT_SERVER_DIR/connection.json'" > /tmp/mt-agentrisk-connection.json
MT_NODE=$(python3 -c 'import json; print(json.load(open("/tmp/mt-agentrisk-connection.json"))["host"])')
MT_PORT=$(python3 -c 'import json; print(json.load(open("/tmp/mt-agentrisk-connection.json"))["port"])')
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L "127.0.0.1:53241:$MT_NODE:$MT_PORT" "$MT_CLUSTER"
```

Leave that terminal open. If local port 53241 is occupied by the old probe tunnel,
close that old tunnel first. If Ubuntu SSH cannot reach the cluster, stop here
and diagnose WSL/VPN routing; Docker or model changes will not fix that route.

In your benchmark Ubuntu terminal:

```bash
cd ~/benchmarks/mt-agentrisk/ToolShield
export MT_AGENT_BASE_URL=http://127.0.0.1:53241/v1
MT_JOB_ID=$(python3 -c 'import json; print(json.load(open("/tmp/mt-agentrisk-connection.json"))["job_id"])')
export MT_AGENT_API_KEY="$(ssh darrius.ng.2024@10.193.104.102 "cat /common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS/outputs/mt_agentrisk_server/$MT_JOB_ID/api_token")"
curl --fail http://127.0.0.1:53241/health
.venv/bin/python local_setup/test_remote_tools.py --method vdcgs \
  --output "smoke_vdcgs_${MT_JOB_ID}.json"
```

The health response must identify `mt-agentrisk-tools-v1` and `vdcgs`.
The old `/generate` text-only service is incompatible. The smoke makes two real
inference calls: emit a harmless tool action, then consume a synthetic tool result
and emit `finish`. It executes no generated command. It must report `passed: true`.
Do not discard a failed smoke and keep resampling: retain the JSON and server
audits for diagnosis.

## User-requested real-task diagnostic

A real-task diagnostic is now explicitly authorized despite the synthetic
probe failure. Use `--diagnostic-task-smoke` for local-method `multi.71` only,
with the recorded probe from the same live job. The failed outcome is preserved
in launch.json; it is never relabelled a pass. See [the real-task smoke guide](MT_AGENTRISK_REAL_TASK_SMOKE.md) for exact
job260035 commands. Standard runs retain the passing-probe gate.

## Run one benchmark task after the smoke passes

Keep your existing `OPENROUTER_API_KEY` set for the benchmark environment helper.
That helper remains Claude Sonnet 4.5 independently of the local agent method.

```bash
.venv/bin/python local_setup/run_baseline.py --method vdcgs --task-id multi.71 \
  --smoke-report "smoke_vdcgs_${MT_JOB_ID}.json" --execute
```

This actually runs tools in Docker and may make paid environment-helper calls.
The launcher checks that the smoke belongs to the current server job and method.
It refuses existing output directories. The default output is
`output/vdcgs/multi.71`; use `--output-dir output/vdcgs/multi.71_retry_02` only for
an investigated retry, preserving the failed run. The launcher creates the output
directory after preflight checks. Do not create it yourself.

If shared-workspace access fails again, use the named-user/default ACL commands
in `local_setup/README.md`; do not run the evaluator with sudo.

Review the trajectory, final state, server audit, critic context, runtime and GPU
memory before expanding beyond this Terminal smoke. A finished controller is not
an official safety score. Post-evaluation judging remains a separate step.

## Other methods

| Method | Inference | Launcher option |
| --- | --- | --- |
| VDCGS | Zephyr plus existing trained HL/LL critics | `--method vdcgs` |
| RDCGS | Same plus existing regret policy | `--method rdcgs` |
| SmoothLLM | Zephyr, 8 copies, existing 10% perturbation policy | `--method smoothllm` |
| TPO | Zephyr plus existing reward model, 5 samples and 2 iterations | `--method tpo` |
| Zephyr control | Single undefended Zephyr response | `--method zephyr` |
| GPT-4o | Native OpenRouter tools | `--method gpt4o` |
| Original baseline | Existing Claude Sonnet 4.5 profile | `--method official` |

For RDCGS, SmoothLLM, TPO or Zephyr, submit the same Slurm script with that method
as its argument, reconnect to the new node/port/token, run the matching protocol
smoke, then pass the matching method and smoke report to the launcher. Each
method uses a separate output folder. Only one benchmark task may use the shared
Docker workspace at a time. TPO performs many model calls per action and may be
substantially slower. GPU memory/time limits still need measurement.

GPT-4o needs no cluster job or tunnel:

```bash
.venv/bin/python local_setup/run_baseline.py --method gpt4o --task-id multi.71
.venv/bin/python local_setup/run_baseline.py --method gpt4o --task-id multi.71 --execute
```

Its agent credential is always `OPENROUTER_API_KEY`, even if a cluster token remains
in `MT_AGENT_API_KEY`. Provider/environment calls are paid. `MT_ENV_API_KEY` and
`MT_ENV_MODEL` may separately configure the environment helper. The GPT-4o profile
uses native function tools and the official 16,384-token output ceiling; local
profiles retain their existing method budgets. These are method configurations,
not a token-budget-matched comparison.

## Live smoke update: job 259882

The first A100 VDCGS service loaded correctly on albert. Its first real tool
request failed HTTP422: all five selected-pool candidates were prose describing
the action, not JSON actions. No tool executed and the second smoke request was
not sent. GPU policy replay passed; inference took11.71s with13.92GiB peak allocated.
The initial LL adapter used a system-only chat template followed by raw text.
The revised `dcgs-tool-candidates-v2` prompt renders complete system/user/assistant
boundaries and repeats the JSON-per-candidate requirement next to the original
list instructions. Nine regression tests and a real-tokenizer/replayed-response
check pass. These tests do not establish new GPU generation success. The running
job retains the old loaded code; user must cancel259882 and submit a fresh A100
job before retesting. Preserve the failed job audits. No Windows code changed.

## Experimental boundary and current evidence

The cluster service emits actions; only the Windows Docker environment executes
them. Local methods receive the same versioned text serialization of the current
tool schemas and actual conversation supplied by OpenHands. Images are rejected.
OpenHands' own message formatting/condensation settings remain unchanged.

DCGS constructs its belief state anew from that supplied transcript for each
action. Its LL generation receives an additional transcript/schema prefix; the
existing belief selection, trained critic scoring, regret calculation and
five-candidate selection stay in place. Each candidate must contain a JSON action
inside the original candidate-list wrappers. This is an explicit tool-agent
adaptation, not a claim of unchanged benchmark methodology or trained tool use.

The selected response is parsed only after method selection. Invalid JSON,
unknown tools and schema-invalid arguments are recorded as failures. There is no
adapter repair, valid-candidate filtering, alternate-method fallback or extra
resampling. The underlying defense's own existing internal behavior is preserved.
HTTP retries reuse both successful and failed results. A fresh benchmark run gets
a new run identifier so independent experiments do not share cached generations.

SmoothLLM perturbs only the latest real user text; tool schemas and observations
are preserved. Its existing refusal heuristic sees raw candidate JSON. TPO scores
the serialized transcript and raw candidate action using its existing reward
model. These critics/rewards have not been validated on tool actions.

Actor input overflow is rejected. DCGS retains its original HL critic 1,500-token
left truncation, with diagnostics, and LL critic 8,192-token hard limit. The saved
Claude trajectory reconstructed without OpenHands recall messages ranges from
6,579 to 7,745 tokens. Conservative LL budget estimates flag the last three of six
decisions as possible overflows (maximum 8,516); short synthetic candidates fit.
Actual runtime checks decide. Longer DCGS traces may fail earlier. No conversation
is silently trimmed by this adapter to make the LL critic fit.

Offline policy/protocol tests, all five artifact/configuration checks, six launcher
tests, and a local HTTP round trip using LiteLLM 1.74.3/OpenAI 1.99.9 passed.
HTTP tests used synthetic model responses. Real GPU action generation and native
Docker execution for the new profiles remain to be tested. The prior successful
Claude `multi.71` run establishes only the original Terminal environment baseline.
