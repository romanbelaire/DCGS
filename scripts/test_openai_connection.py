#!/usr/bin/env python3
"""Test OpenAI-compatible API connectivity without generating tokens."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)


PROXY_VARIABLES = (
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "ALL_PROXY",
    "https_proxy",
    "http_proxy",
    "all_proxy",
)


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without replacing exported values."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and not os.environ.get(key):
            os.environ[key] = os.path.expandvars(value)


def print_exception_chain(exc: BaseException) -> None:
    current: BaseException | None = exc
    depth = 0
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        prefix = "Error" if depth == 0 else "Caused by"
        print(f"{prefix}: {type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
        depth += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check DNS/TLS/API authentication through the OpenAI Models API. "
            "This does not submit a generation request or consume model tokens."
        )
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Environment file to load if variables are not exported (default: .env).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model ID to look for in the response (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Request timeout in seconds (default: 20).",
    )
    parser.add_argument(
        "--ignore-environment-proxy",
        action="store_true",
        help="Run HTTPX with trust_env=False, ignoring all inherited proxy settings.",
    )
    return parser.parse_args()


def validate_proxy_environment() -> bool:
    """Report proxy shape without printing credentials or the full URL."""
    valid = True
    for name in PROXY_VARIABLES:
        value = os.getenv(name, "").strip()
        if not value:
            continue
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https", "socks5", "socks5h"}:
            print(
                f"FAIL: {name} is set but has no supported URL scheme. "
                "It must start with http://, https://, socks5://, or socks5h://."
            )
            valid = False
            continue
        print(
            f"  Proxy: {name} is set "
            f"(scheme={parsed.scheme}, host={parsed.hostname or '<missing>'})"
        )
    return valid


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()

    print("OpenAI-compatible API connection test")
    key_state = f"set ({len(api_key)} characters)" if api_key else "missing"
    print(f"  Key: {key_state}")
    print(f"  Endpoint: {base_url or 'https://api.openai.com/v1'}")

    if base_url:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            print("FAIL: OPENAI_BASE_URL is not a valid HTTP(S) URL.")
            return 1

    if not api_key:
        print("FAIL: Set OPENAI_API_KEY in .env or export it in the shell.")
        return 1

    if not validate_proxy_environment():
        print(
            "Fix the malformed proxy URL or temporarily unset the proxy variables, "
            "then run this test again."
        )
        return 9

    client_kwargs = {
        "api_key": api_key,
        "base_url": base_url or "https://api.openai.com/v1",
        "timeout": args.timeout,
        "max_retries": 0,
    }

    http_client = None
    if args.ignore_environment_proxy:
        print("  HTTP environment discovery: disabled (trust_env=False)")
        http_client = httpx.Client(timeout=args.timeout, trust_env=False)
        client_kwargs["http_client"] = http_client

    try:
        client = OpenAI(**client_kwargs)
        print(f"  Effective client base URL: {client.base_url}")
        models = client.models.list()
        model_ids = {model.id for model in models.data}
    except AuthenticationError as exc:
        print("FAIL: The endpoint was reached, but it rejected the API key.")
        print_exception_chain(exc)
        return 2
    except PermissionDeniedError as exc:
        print("FAIL: The endpoint was reached, but this key lacks permission.")
        print_exception_chain(exc)
        return 3
    except RateLimitError as exc:
        print("FAIL: The endpoint was reached, but it returned a rate-limit response.")
        print_exception_chain(exc)
        return 4
    except APITimeoutError as exc:
        print("FAIL: The connection timed out.")
        print_exception_chain(exc)
        return 5
    except APIConnectionError as exc:
        print("FAIL: DNS, TCP, TLS, proxy, or routing prevented the connection.")
        print_exception_chain(exc)
        return 6
    except APIStatusError as exc:
        print(
            f"FAIL: The endpoint was reached but returned HTTP {exc.status_code}. "
            "Some compatible gateways do not implement the Models API."
        )
        print_exception_chain(exc)
        return 7
    except Exception as exc:
        print("FAIL: Unexpected client error.")
        print_exception_chain(exc)
        return 8
    finally:
        if http_client is not None:
            http_client.close()

    print(f"PASS: Authenticated connection succeeded; endpoint returned {len(model_ids)} models.")
    if args.model in model_ids:
        print(f"PASS: Requested model is listed: {args.model}")
    else:
        print(f"WARNING: {args.model!r} was not present in the returned model list.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
