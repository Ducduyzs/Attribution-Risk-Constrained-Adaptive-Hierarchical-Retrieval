"""Read-only machine/provenance probe; never print credentials."""
import json
import os
from pathlib import Path

root = Path('/workspace/edahr')
config = root / 'artifacts/baselines/main/config.json'
data = json.loads(config.read_text()) if config.exists() else {}
print(json.dumps({
    'config_keys': sorted(data),
    'api_key_present': bool(data.get('openai_api_key') or os.getenv('OPENAI_API_KEY')),
    'raptor': {k: v for k, v in (data.get('raptor_faithful') or {}).items() if 'key' not in k},
    'context_files': [str(p) for p in (root / 'artifacts/contextual').rglob('*') if p.is_file()],
}, indent=2))
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
