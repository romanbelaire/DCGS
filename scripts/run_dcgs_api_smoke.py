#!/usr/bin/env python3
"""Bounded prompt-only DCGS HTTP smoke; not a tool-agent/model-API adapter."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import socket
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
SMOKE_PROMPT = "Explain what a GPU does in two short sentences."


def request_payload(payload):
    if not isinstance(payload, dict) or set(payload) - {"prompt", "seed", "request_id"}:
        raise ValueError("Only prompt, seed, request_id are supported; no history/tools yet")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4096:
        raise ValueError("prompt must contain 1 to 4096 characters")
    seed = payload.get("seed", 0)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request_id):
        raise ValueError("request_id must contain 1 to 64 letters, digits, underscores or hyphens")
    return {"prompt": prompt, "seed": seed, "request_id": request_id}


def probe_row(prompt):
    # No reference answer is supplied or read. This endpoint serves one prompt.
    return {"id": 1, "history": [{"user": prompt, "bot": ""}]}


def save_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def audit_summary(result, config, peak_bytes):
    from safedial_dcgs_run_state import context_summary
    events = result["dcgs_original"]["events"]
    ll_events = [e for e in events if e["request"]["kind"] == "score_ll"]
    if not ll_events:
        raise RuntimeError("No trained LL critic scoring recorded")
    return {"generation_calls": sum(e["request"]["kind"] == "generate" for e in events),
            "ll_critic_calls": len(ll_events),
            "ll_candidates_scored": sum(len(e["request"]["actions"]) for e in ll_events),
            "regret_enabled": config.use_regret_critic,
            "peak_gpu_allocated_bytes": peak_bytes,
            "critic_context": context_summary([
                {**e, "dialogue_id": 1, "turn_index": 0}
                for e in events if e["request"]["kind"] == "score"])}


class Service:
    def __init__(self, infer, token, output, method, max_requests=3):
        self.infer, self.token, self.output = infer, token, output
        self.method, self.max_requests = method, max_requests
        self.results = {}
        self.attempts = 0
        self.fatal = False

    def health(self):
        return {"ok": not self.fatal, "service": "dcgs-api-smoke",
                "ready": not self.fatal, "host": socket.gethostname(), "method": self.method,
                "gpu_models_loaded": True, "tool_agent_adapter": False,
                "attempts": self.attempts, "max_requests": self.max_requests}

    def generate(self, authorization, payload):
        if not isinstance(authorization, str) or not secrets.compare_digest(
                authorization.encode(), ("Bearer " + self.token).encode()):
            return 401, {"error": "Bearer token required"}
        try:
            request = request_payload(payload)
        except ValueError as exc:
            return 400, {"error": str(exc)}
        key = request["request_id"]
        if key in self.results:
            previous, status, response = self.results[key]
            if previous != request:
                return 409, {"error": "request_id was already used for another payload"}
            return status, response
        if self.fatal:
            return 503, {"error": "Service stopped accepting inference after an engine failure"}
        if self.attempts >= self.max_requests:
            return 429, {"error": "Smoke inference request limit reached"}
        self.attempts += 1
        started = time.perf_counter()
        save_json(self.output / f"{key}.request.json", request)
        try:
            result = self.infer(request["prompt"], request["seed"])
            response = {"ok": True, "request_id": key, "method": self.method,
                        "message": result["message"], "inference": True,
                        "elapsed_seconds": time.perf_counter() - started,
                        "audit_summary": result["audit_summary"]}
            save_json(self.output / f"{key}.audit.json", result)
            status = 200
        except Exception as exc:
            self.fatal = True
            response = {"ok": False, "request_id": key, "method": self.method,
                        "error": type(exc).__name__, "details": str(exc),
                        "elapsed_seconds": time.perf_counter() - started,
                        "automatic_retry": False}
            if hasattr(exc, "audit"):
                save_json(self.output / f"{key}.failed_audit.json", exc.audit)
            status = 500
        save_json(self.output / f"{key}.response.json", response)
        self.results[key] = (request, status, response)
        return status, response


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(15)

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
            # Saved result remains cached if the client connection was lost.
            pass

    def do_GET(self):
        if self.path == "/health":
            status = 503 if self.server.service.fatal else 200
            self.reply(status, self.server.service.health())
        else:
            self.reply(404, {"error": "Use GET /health or POST /generate"})

    def do_POST(self):
        if self.path != "/generate":
            return self.reply(404, {"error": "Unknown endpoint"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 32768:
                return self.reply(413, {"error": "JSON body must be 1 to 32768 bytes"})
            payload = json.loads(self.rfile.read(size))
        except (ValueError, UnicodeDecodeError, TimeoutError):
            return self.reply(400, {"error": "Invalid or incomplete JSON request"})
        status, value = self.server.service.generate(self.headers.get("Authorization"), payload)
        self.reply(status, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["vdcgs", "rdcgs"], default="vdcgs")
    parser.add_argument("--artifact-lock", type=Path, default=ROOT / "configs/safedial/dcgs_main.lock.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--duration", type=int, default=1800)
    parser.add_argument("--max-requests", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.duration <= 3300 or not 1 <= args.max_requests <= 10:
        parser.error("duration must be 1..3300 seconds and max-requests 1..10")
    if not 0 <= args.port <= 65535:
        parser.error("port must be 0..65535 (0 chooses an available port)")
    if not args.validate_only and not os.environ.get("SLURM_JOB_ID"):
        parser.error("Real GPU inference requires a user-submitted Slurm allocation")
    os.umask(0o077)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    from run_safedial_dcgs_wildjailbreak import (
        load_artifact_lock, verify_artifacts, tokenizer_for, source_hashes, preflight)
    from safedial_dcgs_wildjailbreak import configuration, LocalBackend, generate_turn, validate_turn
    lock = load_artifact_lock(args.artifact_lock)
    verify_artifacts(lock)
    config = configuration(args.method, "cuda:0", lock)
    tokenizer = tokenizer_for(lock)
    manifest = {"protocol": "dcgs_prompt_only_api_smoke_v1", "method": args.method,
                "source_sha256": source_hashes(), "artifacts": lock,
                "service_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "effective_config": vars(config), "tool_agent_adapter": False,
                "max_requests": args.max_requests, "automatic_retry": False}
    save_json(output / "manifest.json", manifest)
    report = preflight([probe_row(SMOKE_PROMPT)], config, tokenizer)
    save_json(output / "preflight.json", report)
    if args.validate_only:
        print(json.dumps({"passed": report["passed"], "model_calls": 0,
                          "method": args.method, "output_dir": str(output)}), flush=True)
        return 0

    backend = LocalBackend(config, lock)
    save_json(output / "model_loading.json", backend.loading)

    def infer(prompt, seed):
        # Prompt-specific synthetic capacity scan before any model calls.
        capacity = preflight([probe_row(prompt)], config, backend.tokenizer)
        torch = backend.torch
        torch.cuda.reset_peak_memory_stats(config.device)
        with torch.inference_mode():
            result = generate_turn(probe_row(prompt), 0, seed, config, backend.tokenizer, backend)
        # The existing replay validator expects a persisted JSON record (lists,
        # not Python tuples in the original policy's internal audit).
        result = json.loads(json.dumps(result))
        validate_turn({**result, "turn_index": 0, "seed": seed,
                       "generated_response": result["message"]}, probe_row(prompt), config, backend.tokenizer)
        summary = audit_summary(result, config, torch.cuda.max_memory_allocated(config.device))
        summary["policy_replay_passed"] = True
        return {**result, "audit_summary": summary, "preflight": capacity}

    token = secrets.token_urlsafe(32)
    (output / "api_token").write_text(token + "\n")
    service = Service(infer, token, output, args.method, args.max_requests)
    # Serial requests preserve global seed and upstream patching semantics.
    with HTTPServer((args.bind, args.port), Handler) as server:
        server.service = service
        connection = {"host": socket.gethostname(), "port": server.server_port,
                      "method": args.method, "ready": True, "inference": "real GPU",
                      "expires_in_seconds": args.duration, "job_id": os.environ["SLURM_JOB_ID"],
                      "output_dir": str(output), "token_file": str(output / "api_token")}
        save_json(output / "connection.json", connection)
        print("DCGS_API_READY " + json.dumps(connection), flush=True)
        timer = threading.Timer(args.duration, server.shutdown)
        timer.daemon = True
        timer.start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            timer.cancel()
            save_json(output / "server_finished.json", {"attempts": service.attempts, "fatal": service.fatal})
    return int(service.fatal)


if __name__ == "__main__":
    raise SystemExit(main())
