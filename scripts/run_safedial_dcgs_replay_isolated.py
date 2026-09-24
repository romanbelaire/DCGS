#!/usr/bin/env python3
"""Run the unchanged main-policy entrypoints with isolated replay evidence."""
import hashlib
from pathlib import Path
import runpy
import sys

from safedial_dcgs_replay_isolation import isolated_replay


def main():
    scripts = Path(__file__).resolve().parent
    entrypoints = {
        "run": "run_safedial_dcgs_wildjailbreak.py",
        "validate": "validate_safedial_dcgs_wildjailbreak.py",
    }
    if len(sys.argv) < 2 or sys.argv[1] not in entrypoints:
        raise SystemExit("Usage: run_safedial_dcgs_replay_isolated.py {run|validate} [original arguments]")
    target = scripts / entrypoints[sys.argv[1]]
    for name in (Path(__file__).name, "safedial_dcgs_replay_isolation.py"):
        print(f"replay_isolation_sha256 {name} {hashlib.sha256((scripts / name).read_bytes()).hexdigest()}", flush=True)
    sys.argv = [str(target), *sys.argv[2:]]
    with isolated_replay():
        runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
