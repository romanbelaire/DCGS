"""Read-only integrity check for this cleanup; no model/API/scheduler calls."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ARCHIVE = Path(__file__).resolve().parent
ROOT = ARCHIVE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    manifest = json.loads((ARCHIVE / "MANIFEST.json").read_text())
    for rel, expected in manifest["snapshot_sha256"].items():
        assert digest(ARCHIVE / "snapshot" / rel) == expected, ("snapshot changed", rel)
    for rel, expected in manifest["protected_source_sha256"].items():
        assert digest(ROOT / rel) == expected, ("pinned source changed", rel)
    for rel, entry in manifest["run_manifests"].items():
        assert digest(ARCHIVE / entry["archive_path"]) == entry["sha256"], rel
        assert digest(ROOT / rel) == entry["sha256"], ("run config changed", rel)
    retired = manifest["retired_scripts"] + manifest["retired_launchers"] + manifest["retired_tests"]
    assert all(not (ROOT / rel).exists() for rel in retired), "Retired file reappeared"
    for rel, status in manifest["script_inventory"].items():
        if status == "retained":
            assert digest(ROOT / rel) == manifest["snapshot_sha256"][rel], ("retained script changed", rel)
    retired_modules = {Path(rel).stem for rel in manifest["retired_scripts"]}
    for path in list((ROOT / "scripts").glob("*.py")) + list((ROOT / "tests").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) else
                       [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            assert not any(m and m.split(".")[0] in retired_modules for m in modules), (path, modules)
    launchers = sorted((ROOT / "scripts/slurm").glob("*.sbatch"))
    for path in launchers:
        subprocess.run(["bash", "-n", str(path)], check=True)
        for reference in re.findall(r"scripts/[\w/]+\.py", path.read_text()):
            assert (ROOT / reference).is_file(), (path, reference)
    return {"passed": True, "archived_snapshot_files": len(manifest["snapshot_sha256"]),
            "run_manifest_copies": len(manifest["run_manifests"]),
            "unchanged_pinned_sources": len(manifest["protected_source_sha256"]),
            "archived_python_scripts": len(manifest["retired_scripts"]),
            "archived_launchers": len(manifest["retired_launchers"]),
            "archived_test_modules": len(manifest["retired_tests"]),
            "active_python_scripts": len(list((ROOT / "scripts").glob("*.py"))),
            "active_launchers_syntax_and_references_passed": len(launchers)}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
