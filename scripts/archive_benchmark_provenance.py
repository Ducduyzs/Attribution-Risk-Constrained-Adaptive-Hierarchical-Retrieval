"""Archive benchmark provenance as hashes and versions without copying secrets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def file_record(path: Path) -> dict:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def git_output(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = [Path(item) for item in args.input]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing provenance inputs:\n" + "\n".join(missing))

    status = git_output("status", "--porcelain")
    payload = {
        "schema_version": 1,
        "label": args.label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [file_record(path) for path in paths],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions([
                "edahr-neural", "numpy", "torch", "transformers",
                "scikit-learn", "joblib", "openai", "google-genai",
            ]),
        },
        "git": {
            "commit": git_output("rev-parse", "HEAD"),
            "dirty": bool(status),
            "status_porcelain_sha256": (
                hashlib.sha256(status.encode("utf-8")).hexdigest() if status else None
            ),
        },
        "security_note": "Configuration files are represented by hashes only; no secret values are copied.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "files": len(paths)}, indent=2))


if __name__ == "__main__":
    main()