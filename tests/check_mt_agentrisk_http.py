"""Opt-in local HTTP test with the pinned benchmark client; no model/tool calls."""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import urllib.error
import urllib.request
import importlib.util
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / ".cache/mt_agentrisk_client")]
from litellm import completion
from serve_mt_agentrisk import Handler, HTTPServer, Service


def main():
    calls = []
    def engine(request, on_event):
        calls.append(request)
        name = request["tool_choice"]["function"]["name"]
        arguments = {"command": "printf OK"} if name == "execute_bash" else {"message": "OK"}
        return {"message": json.dumps({"type": "tool", "name": name, "arguments": arguments}),
                "prompt_tokens": 10, "completion_tokens": 8}
    engine.manifest = {"audit_replay": "isolated-event-copy-v1", "candidate_parser": "tool-response-wrappers-v1"}

    with tempfile.TemporaryDirectory() as folder, HTTPServer(("127.0.0.1", 0), Handler) as server:
        server.service = Service(engine, "test-only", folder, "vdcgs")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            for token in (None, 'stale-token'):
                request = urllib.request.Request(base + '/v1/models',
                    headers={} if token is None else {'Authorization': 'Bearer ' + token})
                try:
                    urllib.request.urlopen(request, timeout=15)
                except urllib.error.HTTPError as exc:
                    assert exc.code == 401
                else:
                    raise AssertionError('Model listing accepted missing/stale token')
            with urllib.request.urlopen(urllib.request.Request(base + '/v1/models',
                    headers={'Authorization': 'Bearer test-only'}), timeout=15) as response:
                assert json.load(response)['data'][0]['id'] == 'mt-vdcgs'
            assert not calls
            launcher_path = ROOT.parent / 'mt_agentrisk_transfer/ToolShield/local_setup/run_baseline.py'
            spec = importlib.util.spec_from_file_location('http_test_launcher', launcher_path)
            launcher = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(launcher)
            report = Path(folder) / 'failed_probe.json'
            report.write_text(json.dumps({'passed': False, 'method': 'vdcgs', 'server_job_id': 'http-fixture'}))
            with patch.dict(os.environ, {'SLURM_JOB_ID': 'http-fixture', 'MT_AGENT_API_KEY': 'test-only'}):
                config = launcher.configuration({'MT_AGENT_BASE_URL': base + '/v1'}, 'vdcgs')
                health = launcher.verify_local_service(config, 'vdcgs', report, diagnostic=True)
                assert health['job_id'] == 'http-fixture' and not calls
            malformed = urllib.request.Request(base + '/v1/chat/completions',
                data=b'{"model":"mt-vdcgs","model":"mt-rdcgs"}',
                headers={'Authorization': 'Bearer test-only', 'Content-Type': 'application/json'})
            try:
                urllib.request.urlopen(malformed, timeout=15)
            except urllib.error.HTTPError as exc:
                assert exc.code == 400
            else:
                raise AssertionError('Duplicate request fields accepted')
            assert not calls
            tools = [{"type": "function", "function": {"name": name, "parameters": {
                "type": "object", "properties": {arg: {"type": "string"}}, "required": [arg]}}}
                for name, arg in (("execute_bash", "command"), ("finish", "message"))]
            messages = [{"role": "user", "content": "Print OK"}]
            def call(name):
                return completion(model="openai/mt-vdcgs--http_fixture", messages=messages,
                    api_base=f"http://127.0.0.1:{server.server_port}/v1", api_key="test-only",
                    tools=tools, tool_choice={"type": "function", "function": {"name": name}},
                    max_tokens=1024, temperature=0.0, top_p=1.0, seed=0, timeout=15, num_retries=0)
            first = call("execute_bash")
            retry = call("execute_bash")
            assert first.id == retry.id and len(calls) == 1
            message = first.choices[0].message.model_dump(exclude_none=True)
            assert first.choices[0].finish_reason == "tool_calls"
            assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"command": "printf OK"}
            messages.extend([message, {"role": "tool", "tool_call_id": message["tool_calls"][0]["id"],
                                       "content": "Synthetic observation: OK"}])
            second = call("finish")
            assert second.choices[0].message.tool_calls[0].function.name == "finish"
            assert calls[-1]["messages"][-1]["content"] == "Synthetic observation: OK"
            assert len(calls) == 2 and second.usage.total_tokens == 18
            print(json.dumps({"passed": True, "client": "litellm 1.74.3 / openai 1.99.9",
                              "http_requests": 9, "fixture_inferences": 2, "model_calls": 0,
                              "tool_result_round_trip": True, "retry_cached": True,
                              "authentication_preflight": True, "launcher_preflight": True,
                              "ambiguous_json_rejected": True}))
        finally:
            server.shutdown()
            thread.join()


if __name__ == "__main__":
    main()
