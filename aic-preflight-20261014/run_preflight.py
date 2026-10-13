import concurrent.futures, hashlib, json, pathlib, shlex, subprocess

base=pathlib.Path(__file__).resolve().parent
root=base/'attempt-2'
root.mkdir(exist_ok=False)
source=(base/'probe.py').read_text()
ssh=['ssh','-p','2222','-i','/Users/rei/.ssh/id_ed25519','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=20','-o','ProxyCommand=ssh -p 2222 -i /Users/rei/.ssh/id_ed25519 -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -W %h:%p reiwatanabe@casper.ai.hc.keio.ac.jp','reiwatanabe@172.16.0.18']
commands=[]
for worker in (1,2):
    remote=f'srun --partition=gpu-short --ntasks=1 --gres=gpu:a100:1 --cpus-per-task=1 --mem=2G --time=00:03:00 --job-name=spinlanguage-preflight-{worker}-v2 --unbuffered bash -s'
    commands.append(['rtk','proxy']+ssh+[remote])
manifest=dict(purpose='CUDA infrastructure preflight only; research protocol unavailable',
    source_sha256=hashlib.sha256(source.encode()).hexdigest(),commands=commands,
    max_concurrent_jobs=2,wall_limit_per_job_seconds=180,benchmark_repeats=3,
    output_destination=str(root),estimated_local_output_bytes=100000,
    remote_storage='unique temporary directory on allocated compute node; no /share writes',
    research_execution_authorized_but_pending='current-chat code, inputs, budget, stopping and comparison conditions')
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
script='''set -eu
echo "JOB_ID=$SLURM_JOB_ID HOST=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
ENV=/share/reiwatanabe/ceo-bounded-source-20261007-v1/venv
export CUDA_PATH="$ENV/lib/python3.12/site-packages/nvidia/cuda_runtime"
export LD_LIBRARY_PATH="$CUDA_PATH/lib:$ENV/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export CUPY_ACCELERATORS=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/dev/shm
AVAIL=$(df -B1 --output=avail /dev/shm | tail -1)
test "$AVAIL" -gt 67108864
TASK_TMP=$(mktemp -d /dev/shm/spinlanguage-preflight-${SLURM_JOB_ID}-XXXXXX)
export CUPY_CACHE_DIR="$TASK_TMP/cupy-cache"
echo "SCRATCH=$TASK_TMP"
df -h "$TASK_TMP"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
cat > "$TASK_TMP/probe.py" <<'PROBE_SOURCE'
'''+source+'''PROBE_SOURCE
"$ENV/bin/python" "$TASK_TMP/probe.py"
'''
(root/'submitted_script.sh').write_text(script)
def run(i):
    with (root/f'worker-{i+1}.log').open('w') as log:
        proc=subprocess.run(commands[i],input=script,text=True,stdout=log,stderr=subprocess.STDOUT)
    return dict(worker=i+1,returncode=proc.returncode,log=str(root/f'worker-{i+1}.log'))
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    results=list(pool.map(run,range(2)))
(root/'status.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
