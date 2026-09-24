#!/usr/bin/env python3
"""Run the source-pinned 384-token experiment with private replay evidence."""
import hashlib
from pathlib import Path

import run_safedial_dcgs_belief384 as runner
from safedial_dcgs_replay_isolation import isolated_replay


def main(argv=None):
    scripts = Path(__file__).resolve().parent
    for name in (Path(__file__).name, "safedial_dcgs_replay_isolation.py"):
        print(f"replay_isolation_sha256 {name} {hashlib.sha256((scripts / name).read_bytes()).hexdigest()}", flush=True)
    with isolated_replay():
        return runner.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
