"""Create this project's isolated CUDA dependencies on an allocated GPU node."""
import json, pathlib, subprocess

ROOT=pathlib.Path(__file__).resolve().parents[1]
REMOTE='/dev/shm/rei-magnonic-lm-20261014-v1'
def ssh_command() -> list[str]:
    return json.loads((ROOT/'aic-preflight-20261014/attempt-2/manifest.json').read_text())['commands'][0][:-1]

if __name__=='__main__':
    script=f'''set -eu
echo "SETUP_JOB=$SLURM_JOB_ID HOST=$(hostname)"
test ! -e {REMOTE}
mkdir -m 700 {REMOTE}
export TMPDIR={REMOTE}
df -h /dev/shm
test "$(df -B1 --output=avail /dev/shm | tail -1)" -gt 12884901888
/share/reiwatanabe/ceo-bounded-source-20261007-v1/venv/bin/python -m pip install --no-cache-dir --target {REMOTE}/site torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
export PYTHONPATH={REMOTE}/site
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/share/reiwatanabe/ceo-bounded-source-20261007-v1/venv/bin/python -c 'import torch; print(torch.__version__,torch.version.cuda,torch.cuda.is_available()); print(torch.cuda.get_device_properties(0)); print((torch.ones(1000000,device="cuda")**2).sum().item())'
touch {REMOTE}/environment-ready
du -sh {REMOTE}
'''
    remote='srun --partition=gpu-standard --ntasks=1 --gres=gpu:a100:1 --cpus-per-task=2 --mem=24G --time=00:30:00 --job-name=magnonic-env-setup --unbuffered bash -s'
    cmd=ssh_command()+[remote]
    (ROOT/'results/setup-command.json').write_text(json.dumps(dict(command=cmd,script=script,remote_directory=REMOTE),indent=2))
    with (ROOT/'results/setup.log').open('w') as log:
        result=subprocess.run(cmd,input=script,text=True,stdout=log,stderr=subprocess.STDOUT)
    (ROOT/'results/setup-status.json').write_text(json.dumps(dict(returncode=result.returncode)))
    print(result.returncode)
