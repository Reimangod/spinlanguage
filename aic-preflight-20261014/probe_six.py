import hashlib, json, os, platform, time
import numpy as np
import cupy as cp

count=cp.cuda.runtime.getDeviceCount()
assert count == 6, f'Expected exactly six visible CUDA devices, got {count}'
x=np.linspace(-1,1,2000000,dtype=np.float64)
expected=x*x+np.sin(x)
arrays=[]
streams=[]
for device in range(count):
    with cp.cuda.Device(device):
        stream=cp.cuda.Stream(non_blocking=True)
        with stream:
            a=cp.asarray(x)
            result=a*a+cp.sin(a)
        stream.synchronize()
        arrays.append(a)
        streams.append(stream)
# Launch independently on all allocated GPUs before synchronizing any GPU.
start=time.perf_counter()
results=[]
for device in range(count):
    with cp.cuda.Device(device), streams[device]:
        a=arrays[device]
        for repeat in range(20):
            result=a*a+cp.sin(a)
        results.append(result)
for device in range(count):
    with cp.cuda.Device(device):
        streams[device].synchronize()
elapsed=time.perf_counter()-start
records=[]
for device in range(count):
    with cp.cuda.Device(device):
        props=cp.cuda.runtime.getDeviceProperties(device)
        result=cp.asnumpy(results[device])
        np.testing.assert_allclose(result,expected,rtol=1e-12,atol=1e-12)
        records.append(dict(logical_device=device,name=props['name'].decode(),
            pci_bus_id=cp.cuda.runtime.deviceGetPCIBusId(device),
            memory_bytes=props['totalGlobalMem'],max_abs_error=float(np.max(np.abs(result-expected)))))
assert len({r['pci_bus_id'] for r in records})==6
print(json.dumps(dict(job_id=os.environ['SLURM_JOB_ID'],hostname=platform.node(),
    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),backend='CuPy CUDA float64',
    visible_device_count=count,devices=records,all_devices_passed=True,
    source_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest(),
    input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
    concurrent_launch_seconds=elapsed,repeats_per_device=20,
    purpose='six-GPU allocation and CUDA smoke test; no scientific experiment or speedup claim'),indent=2),flush=True)
