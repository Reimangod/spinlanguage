import hashlib, json, os, platform, time
import importlib.metadata as metadata
import numpy as np
import cupy as cp

def timed(fn, sync=lambda: None):
    times=[]
    for _ in range(3):
        sync(); start=time.perf_counter(); value=fn(); sync()
        times.append(time.perf_counter()-start)
    return min(times), value

cp.cuda.Device(0).use()
props=cp.cuda.runtime.getDeviceProperties(0)
x=np.linspace(-1,1,10001,dtype=np.float64)
gx=cp.asarray(x)
cpu=x*x+np.sin(x)
gpu=cp.asnumpy(gx*gx+cp.sin(gx))
np.testing.assert_allclose(cpu,gpu,rtol=1e-12,atol=1e-12)
# Generic arithmetic check only; this is not the research energy or gradient.
n=2000000
a=np.linspace(-1,1,n,dtype=np.float64)
ga=cp.asarray(a)
fn_cpu=lambda: np.sum(a*a+np.sin(a))
fn_gpu=lambda: cp.sum(ga*ga+cp.sin(ga))
fn_gpu(); cp.cuda.Stream.null.synchronize()
cpu_time,cpu_value=timed(fn_cpu)
gpu_time,gpu_value=timed(fn_gpu,cp.cuda.Stream.null.synchronize)
np.testing.assert_allclose(cpu_value,float(gpu_value.get()),rtol=1e-12,atol=1e-8)
transfer_time,_=timed(lambda: cp.asnumpy(cp.sum(cp.asarray(a)**2+cp.sin(cp.asarray(a)))),cp.cuda.Stream.null.synchronize)
result=dict(job_id=os.environ.get('SLURM_JOB_ID'),hostname=platform.node(),
    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),logical_device=0,
    gpu_name=props['name'].decode(),gpu_memory_bytes=props['totalGlobalMem'],
    cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),cuda_driver=cp.cuda.runtime.driverGetVersion(),
    input_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
    source_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest(),
    versions={p:metadata.version(p) for p in ['numpy','scipy','qiskit','cupy-cuda12x']},
    max_abs_error=float(np.max(np.abs(cpu-gpu))),sum_abs_error=float(abs(cpu_value-float(gpu_value.get()))),
    cpu_seconds=cpu_time,gpu_resident_seconds=gpu_time,gpu_transfer_inclusive_seconds=transfer_time,
    speedup_resident=cpu_time/gpu_time,speedup_transfer_inclusive=cpu_time/transfer_time,
    backend='numpy CPU / cupy CUDA',benchmark='generic float64 elementwise and reduction; not research workload')
print(json.dumps(result,indent=2),flush=True)
