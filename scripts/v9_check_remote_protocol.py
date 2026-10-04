"""Normalize Windows path-key separators ONLY after verifying exact code bytes."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / 'artifacts/v9_results/protocol.json'
data = json.loads(path.read_text(encoding='utf-8'))
canonical = {}
for name, expected in data['code_hashes'].items():
    name = name.replace('\\', '/')
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError('Code fingerprint mismatch: ' + name)
    canonical[str(Path(name))] = actual
data['code_hashes'] = canonical
path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
print('Protocol code bytes verified; path separators normalized for this OS.')
