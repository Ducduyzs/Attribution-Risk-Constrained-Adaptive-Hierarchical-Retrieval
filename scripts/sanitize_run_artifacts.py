"""Remove credentials from JSON run artifacts before syncing or publishing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SENSITIVE_KEYS = {
    "api_key", "apikey", "access_token", "refresh_token", "auth_token",
    "bearer_token", "password", "secret", "client_secret", "credential",
    "credentials",
}


def is_sensitive(key: object) -> bool:
    lowered = str(key).lower()
    return (
        lowered in SENSITIVE_KEYS
        or lowered.endswith("_api_key")
        or lowered.endswith("_password")
        or lowered.endswith("_secret")
    )


def redact(value):
    if isinstance(value, dict):
        return {
            key: ("<redacted>" if item else None) if is_sensitive(key)
            else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--restore-settings-in", type=Path)
    args = parser.parse_args()

    if args.config and args.restore_settings_in:
        settings = redact(json.loads(args.config.read_text(encoding="utf-8")))
        target = args.restore_settings_in
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload["settings"] = settings
        target.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    changed = 0
    for path in sorted(args.directory.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        path.write_text(json.dumps(redact(payload), indent=1), encoding="utf-8")
        changed += 1
    print(json.dumps({"json_files_sanitized": changed}))


if __name__ == "__main__":
    main()
