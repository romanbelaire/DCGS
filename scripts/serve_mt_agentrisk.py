#!/usr/bin/env python3
"""Authenticated, bounded chat-completions server for one local defense method."""
import argparse
import hashlib
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import secrets
import socket
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache/mt_agentrisk_python"))
from mt_agentrisk_protocol import LOCAL_METHODS, PROTOCOL, ProtocolError, normalize, digest, decode_action, strict_json_loads
from run_dcgs_api_smoke import save_json


def usage_for(result):
    if "prompt_tokens" in result:
        prompt, completion = result["prompt_tokens"], result["completion_tokens"]
    else:
        raw = [g for e in result.get("dcgs_original", {}).get("events", [])
               for g in e.get("result", {}).get("raw_generations", [])]
        prompt = sum(len(ids) for g in raw for ids in g["input_ids"])
        completion = sum(len(ids) for g in raw for ids in g["generated_token_ids"])
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


class Service:
    def __init__(self, engine, token, output, method, max_requests=40):
        self.engine, self.token, self.output, self.method = engine, token, Path(output), method
        self.max_requests, self.attempts, self.results = max_requests, 0, {}

    def authorized(self, value):
        return isinstance(value, str) and secrets.compare_digest(value.encode(), ("Bearer " + self.token).encode())

    def health(self):
        return {"ok": True, "service": PROTOCOL, "method": self.method, "host": socket.gethostname(),
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "attempts": self.attempts, "max_requests": self.max_requests, "tools_execute_here": False,
                "preflight_auth": "models-bearer-v1",
                "audit_replay": getattr(self.engine, "manifest", {}).get("audit_replay"),
                "candidate_parser": getattr(self.engine, "manifest", {}).get("candidate_parser")}

    def complete(self, authorization, payload):
        if not self.authorized(authorization):
            return 401, {"error": {"message": "Bearer token required", "type": "authentication_error"}}
        try:
            request = normalize(payload, self.method)
            key = digest(request)
        except Exception as exc:
            return 400, {"error": {"message": str(exc), "type": "invalid_request_error"}}
        if key in self.results:
            return self.results[key]
        if self.attempts >= self.max_requests:
            return 429, {"error": {"message": "Smoke request budget exhausted", "type": "budget_exhausted"}}
        self.attempts += 1
        folder = self.output / key
        folder.mkdir(exist_ok=False)
        save_json(folder / "request.json", request)
        started = time.perf_counter()
        def event(value):
            with (folder / "events.jsonl").open("a") as f:
                f.write(json.dumps(value) + "\n")
                f.flush()
                os.fsync(f.fileno())
        try:
            result = self.engine(request, event)
            save_json(folder / "audit.json", result)
            message, reason = decode_action(result["message"], request, key)
            response = {"id": "chatcmpl-" + key[:24], "object": "chat.completion", "created": int(time.time()),
                        "model": request["model"], "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                        "usage": usage_for(result), "mt_agentrisk": {"protocol": PROTOCOL, "request_id": key,
                            "method": self.method, "elapsed_seconds": time.perf_counter() - started,
                            "audit_summary": result.get("audit_summary", {})}}
            status = 200
        except Exception as exc:
            if hasattr(exc, "audit"):
                save_json(folder / "failed_audit.json", exc.audit)
            terminal = getattr(exc, "details", {}).get("category") == "terminal_output"
            status = 422 if isinstance(exc, ProtocolError) or terminal else 500
            category = ("model_protocol" if isinstance(exc, ProtocolError) else
                        "terminal_output" if terminal else "runner_error")
            response = {"error": {"message": str(exc), "type": type(exc).__name__, "request_id": key,
                                   "category": category, "automatic_retry": False},
                        "elapsed_seconds": time.perf_counter() - started}
        save_json(folder / "response.json", {"status": status, "body": response})
        self.results[key] = (status, response)
        return status, response


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def reply(self, status, value):
        body = json.dumps(value).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass  # Result is persisted and cached for identical transport retries.

    def do_GET(self):
        if self.path == "/health":
            return self.reply(200, self.server.service.health())
        if self.path == "/v1/models":
            if not self.server.service.authorized(self.headers.get("Authorization")):
                return self.reply(401, {"error": {"message": "Bearer token required"}})
            return self.reply(200, {"object": "list", "data": [
                {"id": "mt-" + self.server.service.method, "object": "model", "owned_by": "local"}]})
        self.reply(404, {"error": {"message": "Unknown endpoint"}})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self.reply(404, {"error": {"message": "Unknown endpoint"}})
        authorization = self.headers.get("Authorization")
        if not self.server.service.authorized(authorization):
            return self.reply(401, {"error": {"message": "Bearer token required"}})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 2_000_000:
                return self.reply(413, {"error": {"message": "Body must be 1..2000000 bytes"}})
            payload = strict_json_loads(self.rfile.read(size))
        except (ValueError, UnicodeDecodeError, TimeoutError):
            return self.reply(400, {"error": {"message": "Invalid JSON"}})
        status, result = self.server.service.complete(authorization, payload)
        self.reply(status, result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=LOCAL_METHODS, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--duration", type=int, default=7200)
    p.add_argument("--max-requests", type=int, default=40)
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args()
    if not 1 <= args.duration <= 43200 or not 1 <= args.max_requests <= 1000 or not 0 <= args.port <= 65535:
        p.error("Invalid duration, budget, or port")
    if not args.validate_only and not os.environ.get("SLURM_JOB_ID"):
        p.error("GPU execution requires a user-submitted Slurm allocation")
    os.umask(0o077)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from mt_agentrisk_engines import Engine
    from run_safedial_dcgs_wildjailbreak import source_hashes
    engine = Engine(args.method, args.validate_only)
    hashes = source_hashes()
    for name in ("mt_agentrisk_protocol.py", "mt_agentrisk_engines.py", "serve_mt_agentrisk.py",
                 "safedial_smoothllm.py", "safedial_tpo_upstream.py", "safedial_tpo_runtime.py"):
        hashes["scripts/" + name] = hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
    for name in ("safedial_tpo.py", "safedial_tpo_artifacts.py", "safedial_generation_retry.py", "run_dcgs_api_smoke.py"):
        hashes["scripts/" + name] = hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
    save_json(args.output_dir / "manifest.json", {"protocol": PROTOCOL, "engine": engine.manifest,
        "source_sha256": hashes, "max_requests": args.max_requests, "validate_only": args.validate_only,
        "selection_then_parse": True, "wire_generation_parameters": "fixed by server method; not client temperature/max_tokens",
        "transport_cache": "full normalized request including per-run model suffix; caches successes and failures"})
    if args.validate_only:
        print(json.dumps({"passed": True, "model_calls": 0, "method": args.method}), flush=True)
        return
    token = secrets.token_urlsafe(32)
    (args.output_dir / "api_token").write_text(token + "\n")
    service = Service(engine, token, args.output_dir, args.method, args.max_requests)
    with HTTPServer((args.bind, args.port), Handler) as server:
        server.service = service
        connection = {"host": socket.gethostname(), "port": server.server_port, "method": args.method,
                      "protocol": PROTOCOL, "job_id": os.environ["SLURM_JOB_ID"],
                      "expires_in_seconds": args.duration, "token_file": str(args.output_dir / "api_token")}
        save_json(args.output_dir / "connection.json", connection)
        print("MT_AGENTRISK_READY " + json.dumps(connection), flush=True)
        timer = threading.Timer(args.duration, server.shutdown)
        timer.daemon = True
        timer.start()
        try:
            server.serve_forever()
        finally:
            timer.cancel()
            save_json(args.output_dir / "finished.json", {"attempts": service.attempts})


if __name__ == "__main__":
    main()
