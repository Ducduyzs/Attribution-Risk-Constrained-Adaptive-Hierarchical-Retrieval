"""Allowlisted v9 artifact bundles. Credentials/configs are never included."""
import argparse
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('direction', choices=['to_gpu', 'from_gpu'])
parser.add_argument('--output', required=True)
args = parser.parse_args()
base = ROOT / 'artifacts/v9_results'
if args.direction == 'to_gpu':
    files = [base / 'protocol.json', base / 'prepared_source.json',
        ROOT / 'src/edahr/experimental_v9.py', ROOT / 'scripts/run_v9_directions.py',
        ROOT / 'scripts/v9_supervise.py', ROOT / 'scripts/v9_sync_bundle.py',
        ROOT / 'scripts/v9_check_remote_protocol.py',
        ROOT / 'tests/test_experimental_v9.py', ROOT / 'analysis/v9_directions_protocol.md']
    files += sorted((base / 'prepared').glob('*.json'))
    files += sorted((base / 'pending').glob('*.json'))
else:
    files = sorted((base / 'rows').glob('*.json'))
    files += [p for p in sorted(base.glob('*.json')) if p.name not in ('protocol.json', 'prepared_source.json')]
    files += sorted(base.glob('*.jsonl')) + sorted(base.glob('*.md'))
files = [p for p in files if p.is_file()]
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(out, 'w:gz') as archive:
    for path in files:
        archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
print(f'{args.direction}: {len(files)} files, {out.stat().st_size} bytes')
