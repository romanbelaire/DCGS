#!/usr/bin/env python3
"""Temporary HTTP reachability probe. No models, files, or tool execution."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import uuid


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def reply(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def identity(self):
        return {"ok": True, "service": "connectivity-probe", "inference": False,
                "host": socket.gethostname(), "probe_id": self.server.probe_id}

    def do_GET(self):
        if self.path != "/health":
            return self.reply(404, {"error": "Use GET /health or POST /echo"})
        self.reply(200, self.identity())

    def do_POST(self):
        if self.path != "/echo":
            return self.reply(404, {"error": "Unknown endpoint"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                return self.reply(413, {"error": "Send 1 to 4096 bytes of JSON"})
            payload = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError):
            return self.reply(400, {"error": "Invalid JSON body or length"})
        self.reply(200, {**self.identity(), "received": payload})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--duration", type=int, default=1800, help="Lifetime in seconds")
    args = parser.parse_args()
    if not 1 <= args.duration <= 3600:
        parser.error("Duration must be between 1 and 3600 seconds")
    with ThreadingHTTPServer((args.bind, args.port), Handler) as server:
        server.probe_id = uuid.uuid4().hex
        timer = threading.Timer(args.duration, server.shutdown)
        timer.daemon = True
        timer.start()
        print(json.dumps({"listening": server.server_address,
                          "host": socket.gethostname(), "probe_id": server.probe_id,
                          "expires_in_seconds": args.duration}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            timer.cancel()
    print("Connectivity probe stopped", flush=True)


if __name__ == "__main__":
    main()
