# DCGS inference API smoke

Test real VDCGS/RDCGS inference from Windows over the existing VPN and SSH
connection before installing the benchmark's Docker environment. The successful
`connectivity-probe` response established only login-node reachability.

This temporary service uses the pinned Zephyr actor, hierarchical critic, and
trained LL critic from `configs/safedial/dcgs_main.lock.json`. It preserves the
existing policy and validates each response through audit replay. It accepts a
single prompt, seed, and request ID; conversation history and tool actions are
not supported yet. This is not an OpenAI-compatible endpoint or a benchmark run.

## Submit on origami

The user submits this command from the DCGS root:

```bash
sbatch scripts/slurm/run_dcgs_api_smoke.sbatch vdcgs
```

It requests one L40S, 64 GB RAM, eight CPUs, and one hour. After artifact checks
and model loading, the server accepts at most three unique requests and stops
accepting connections after 30 minutes; an in-flight request can finish subject
to the job time limit. Repeat with `rdcgs` after the VDCGS test succeeds.

Find `DCGS_API_READY` in `outputs/slurm/dcgs-api-smoke-<job-id>.out`. The chosen
compute hostname and available port are also recorded in
`outputs/dcgs_api_smoke/<job-id>/connection.json`. The file appears only after
the models load. The private bearer token is in the same directory's `api_token`
file, readable only by the owner. Do not paste that token into chat or logs.

## Tunnel from Windows

Keep the institutional VPN connected. In PowerShell, enter the compute hostname
and port from `connection.json`. The local port is selected dynamically because
earlier fixed-port binds on this Windows machine were denied:

```powershell
$computeHost = Read-Host 'Compute hostname from connection.json'
$computePort = [int](Read-Host 'Port from connection.json')
$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
$listener.Start()
$localPort = $listener.LocalEndpoint.Port
$listener.Stop()
Write-Host "Use local port $localPort in the second PowerShell window"
ssh -N -o ExitOnForwardFailure=yes -L "127.0.0.1:${localPort}:${computeHost}:${computePort}" darrius.ng.2024@10.193.104.102
```

Leave that window running. This forwards through origami to the compute node;
the compute-node route has not yet been verified. HTTPS is unnecessary on the
local loopback URL: the Windows-to-origami hop uses SSH. The final
origami-to-compute hop uses HTTP on the cluster network. This temporary service
is not configured for public internet exposure.

In a second PowerShell window:

```powershell
$localPort = [int](Read-Host 'Local port printed by the tunnel window')
$jobId = Read-Host 'Smoke Slurm job ID'
if ($jobId -notmatch '^\d+$') { throw 'Job ID must be numeric' }
$baseUrl = "http://127.0.0.1:$localPort"
Invoke-RestMethod "$baseUrl/health"
$tokenPath = "/common/home/users/d/darrius.ng.2024/projects/RL-Defense/DCGS/outputs/dcgs_api_smoke/$jobId/api_token"
$tokenLines = ssh darrius.ng.2024@10.193.104.102 "cat $tokenPath"
if ($LASTEXITCODE -ne 0) { throw 'Could not read API token over SSH' }
$token = ($tokenLines -join "").Trim()
$body = @{prompt='Explain what a GPU does in two short sentences.'; seed=0; request_id='windows-smoke-1'} | ConvertTo-Json
Invoke-RestMethod "$baseUrl/generate" -Method Post -ContentType 'application/json' -Headers @{Authorization="Bearer $token"} -Body $body -TimeoutSec 600
```

Expect `service=dcgs-api-smoke` and `gpu_models_loaded=true` from health. A
successful generation returns `inference=true`, a message, and `audit_summary`
with trained LL scoring and `policy_replay_passed=true`. GPU memory and full
policy audit are saved in the job output directory. Do not confuse a health
response with a successful generation.

Requests run serially. A health request can wait behind inference. If the client
times out, repeat exactly the same request ID and payload to retrieve the cached
result without generating again. Changed content with the same ID returns 409;
the fourth unique request returns 429. An engine failure saves its audit where
available and stops new inference requests; it does not retry automatically.

## Validation and next steps

Local checks use synthetic model responses for policy replay, authentication,
input rejection, retry caching, limits, and failure handling. `--validate-only`
also checks real checkpoint hashes and the real tokenizer's prompt capacity,
without loading GPU models. Evidence is under
`docs/verification/dcgs_api_smoke_20260920/`.

On 2026-09-20, user-submitted job 258910 passed real GPU loading, cluster HTTP
generation, and Windows-to-compute-node generation through VPN/SSH. The saved
`windows-smoke-1.response.json` reports 14.17 seconds, five LL candidates scored,
and successful policy replay. This confirms the architecture with a temporary
service; WSL/Docker-to-service access remains untested. Next, set up WSL2/Docker
for the official MT-AgentRisk baseline. Connecting DCGS
to the benchmark additionally requires a conversation/tool-action adapter and
validation; this prompt-only test does not provide that adapter.
