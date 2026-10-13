"""Two bounded Slurm workers. Stream scalar logs and checkpoints to the Mac.

This orchestrator never retries an ambiguous allocation. Reconcile squeue/sacct
and recorded IDs before a human starts a new attempt. /share is read-only here.
"""
import argparse,base64,concurrent.futures,hashlib,io,json,pathlib,re,subprocess,tarfile,time
from aic_setup import ROOT,REMOTE,ssh_command

def bundle():
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as tar:
        for directory in ('src','tests','scripts','configs','data'):
            for path in sorted((ROOT/directory).rglob('*')):
                if path.is_file() and '__pycache__' not in str(path):
                    tar.add(path,arcname=str(path.relative_to(ROOT)))
    return buffer.getvalue()

def run_worker(worker:int,phase:str,payload:bytes,run_id:str):
    directory=f'{REMOTE}/{run_id}-worker-{worker}'
    local=ROOT/'results'/run_id/f'worker-{worker}'
    local.mkdir(parents=True,exist_ok=False)
    entry='scripts/physics_preflight.py' if phase=='preflight' else f'scripts/study.py --worker {worker}'
    wall='00:20:00' if phase=='preflight' else '02:00:00'
    remote=f'srun --partition=gpu-strong --ntasks=1 --gres=gpu:a100:1 --cpus-per-task=2 --mem=16G --time={wall} --job-name=magnonic-{phase}-w{worker} --unbuffered bash -s'
    script=f'''set -eu
echo "JOB_ID=$SLURM_JOB_ID HOST=$(hostname) CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-unset}}"
test -f {REMOTE}/environment-ready
mkdir -m 700 {directory}
cd {directory}
export TMPDIR={directory}
export PYTHONPATH={directory}/src:{REMOTE}/site
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MAGNONIC_SOURCE_DIGEST={hashlib.sha256(payload).hexdigest()}
mkdir results
df -h /dev/shm
test "$(df -B1 --output=avail /dev/shm | tail -1)" -gt 1073741824
nvidia-smi --query-gpu=index,uuid,name,memory.total,pci.bus_id --format=csv
echo "MONITOR_BEGIN"
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw --format=csv -l 2 > results/gpu-utilization.csv &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
/share/reiwatanabe/ceo-bounded-source-20261007-v1/venv/bin/python - <<'UNPACK'
import base64,io,tarfile
data=base64.b64decode('{base64.b64encode(payload).decode()}')
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar: tar.extractall('.',filter='data')
UNPACK
/share/reiwatanabe/ceo-bounded-source-20261007-v1/venv/bin/python {entry}
kill "$MONITOR_PID" 2>/dev/null || true
/share/reiwatanabe/ceo-bounded-source-20261007-v1/venv/bin/python - <<'PACK'
import base64,io,tarfile
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w:gz') as t: t.add('results',arcname='results')
print('@@ARTIFACT '+base64.b64encode(b.getvalue()).decode(),flush=True)
PACK
'''
    command=ssh_command()+[remote]
    manifest=dict(command=command,source_bundle_sha256=hashlib.sha256(payload).hexdigest(),
        script_sha256=hashlib.sha256(script.encode()).hexdigest(),directory=directory,
        phase=phase,worker=worker,wall=wall,output_destination=str(local),started_unix=time.time())
    (local/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with (local/'console.log').open('w') as log:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        # Large input is written by a thread, while stdout is drained concurrently.
        def feed():
            try: proc.stdin.write(script); proc.stdin.close()
            except BrokenPipeError: pass
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            feeder=pool.submit(feed)
            for line in proc.stdout:
                if line.startswith('@@CHECKPOINT '):
                    _,name,encoded=line.rstrip().split(' ',2)
                    if pathlib.Path(name).name!=name: raise ValueError('unsafe checkpoint name')
                    (local/name).write_bytes(base64.b64decode(encoded))
                    log.write(f'CHECKPOINT_RECEIVED {name}\n')
                elif line.startswith('@@ARTIFACT '):
                    archive=base64.b64decode(line[len('@@ARTIFACT '):])
                    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as tar:
                        tar.extractall(local,filter='data')
                    log.write(f'ARTIFACT_RECEIVED bytes={len(archive)} sha256={hashlib.sha256(archive).hexdigest()}\n')
                else:
                    log.write(line); log.flush()
                    match=re.search(r'JOB_ID=(\d+)',line)
                    if match:
                        manifest['job_id']=match.group(1)
                        (local/'manifest.json').write_text(json.dumps(manifest,indent=2))
            feeder.result()
        returncode=proc.wait()
    (local/'status.json').write_text(json.dumps(dict(returncode=returncode,ended_unix=time.time())))
    return dict(worker=worker,returncode=returncode,job_id=manifest.get('job_id'),directory=str(local))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',choices=['preflight','study'],required=True)
    args=parser.parse_args()
    payload=bundle(); run_id=f'{args.phase}-{time.strftime("%Y%m%d-%H%M%S")}'
    (ROOT/'results'/run_id).mkdir()
    (ROOT/'results'/run_id/'source.tar.gz').write_bytes(payload)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run_worker,w,args.phase,payload,run_id) for w in (0,1)]
        outcomes=[f.result() for f in futures]
    print(json.dumps(outcomes,indent=2))
