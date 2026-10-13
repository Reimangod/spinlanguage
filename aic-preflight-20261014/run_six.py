import hashlib,json,pathlib,subprocess
base=pathlib.Path(__file__).resolve().parent
root=base/'six-gpu'
root.mkdir(exist_ok=False)
source=(base/'probe_six.py').read_text()
ssh=json.loads((base/'attempt-2/manifest.json').read_text())['commands'][0][:-1]
remote='srun --immediate=30 --partition=gpu-short --ntasks=1 --gres=gpu:a100:6 --cpus-per-task=1 --mem=4G --time=00:03:00 --job-name=spinlanguage-six-gpu-test --unbuffered bash -s'
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
TASK_TMP=$(mktemp -d /dev/shm/spinlanguage-six-${SLURM_JOB_ID}-XXXXXX)
export CUPY_CACHE_DIR="$TASK_TMP/cupy-cache"
echo "SCRATCH=$TASK_TMP"
df -h "$TASK_TMP"
nvidia-smi --query-gpu=name,uuid,pci.bus_id,memory.total --format=csv
cat > "$TASK_TMP/probe.py" <<'PROBE_SOURCE'
'''+source+'''PROBE_SOURCE
"$ENV/bin/python" "$TASK_TMP/probe.py"
'''
(root/'submitted_script.sh').write_text(script)
manifest=dict(command=ssh+[remote],source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    script_sha256=hashlib.sha256(script.encode()).hexdigest(),gpu_count=6,max_jobs=1,
    wall_limit_seconds=180,allocation_wait_limit_seconds=30,repeats_per_device=20,
    output_destination=str(root),research_budget_used=0,purpose='user-authorized six-GPU infrastructure test')
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
with (root/'job.log').open('w') as log:
    result=subprocess.run(ssh+[remote],input=script,text=True,stdout=log,stderr=subprocess.STDOUT)
(root/'status.json').write_text(json.dumps(dict(returncode=result.returncode),indent=2))
print((root/'job.log').read_text())
print(f'RETURN_CODE={result.returncode}')
