"""Install/restart the scoped one-shot v9 experiment (no public service)."""
import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--smoke', action='store_true')
parser.add_argument('--prepare', action='store_true')
parser.add_argument('--verify', action='store_true')
parser.add_argument('--output', default='artifacts/v9_results')
args = parser.parse_args()
root = Path('/workspace/edahr')
name = 'edahr_v9_smoke' if args.smoke else 'edahr_v9'
log = root / (name + '.log')
command = '/venv/main/bin/python -u scripts/run_v9_directions.py'
if args.smoke:
    command += ' --questions 2 --budgets 1024 --output artifacts/v9_smoke'
elif args.prepare:
    command += ' --phase prepare'
elif args.verify:
    command += ' --phase verify --output ' + args.output
else:
    raise SystemExit('Use --verify: generation runs locally so API credentials never leave the local machine.')
config = Path('/etc/supervisor/conf.d') / (name + '.conf')
config.write_text(f'''[program:{name}]
directory={root}
command={command}
environment=PYTHONUNBUFFERED="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMBA_NUM_THREADS="1",TOKENIZERS_PARALLELISM="false"
autostart=false
autorestart=false
startsecs=0
startretries=0
redirect_stderr=true
stdout_logfile={log}
stdout_logfile_maxbytes=20000000
stdout_logfile_backups=2
stopasgroup=true
killasgroup=true
''')
subprocess.run(['supervisorctl', 'reread'], check=True)
subprocess.run(['supervisorctl', 'update', name], check=True)
subprocess.run(['supervisorctl', 'start', name], check=True)
print(log)
