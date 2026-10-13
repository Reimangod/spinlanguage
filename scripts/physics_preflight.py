"""Tests, timestep study, actual workload CUDA gradient and CPU timing gate."""
import json,pathlib,time,subprocess,sys
from dataclasses import replace
import torch
from magnonic.config import PhysicsConfig
from magnonic.physics import YIGReservoir

def timed(r,tokens,device):
    def sync():
        if device=='cuda': torch.cuda.synchronize()
    timings=[]
    for _ in range(3):
        r.zero_grad(set_to_none=True); sync(); start=time.perf_counter()
        values,state=r(tokens); loss=values.square().mean(); loss.backward(); sync()
        timings.append(time.perf_counter()-start)
    return min(timings),values.detach().cpu(),r.raw_current.grad.detach().cpu(),state[0].detach().cpu()

if __name__=='__main__':
    torch.set_num_threads(1)
    root=pathlib.Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'-m','pytest','-q','tests'],cwd=root,capture_output=True,text=True)
    print(result.stdout+result.stderr,flush=True)
    if result.returncode: raise SystemExit(result.returncode)
    torch.manual_seed(123)
    c=PhysicsConfig()
    x=torch.randint(10,(512,32))
    cpu=YIGReservoir(10,c,7)
    gpu=YIGReservoir(10,c,7).cuda(); gpu.load_state_dict(cpu.state_dict())
    gpu(x.cuda())[0].sum().backward(); torch.cuda.synchronize()
    cpu_t,cpu_y,cpu_g,cpu_state=timed(cpu,x,'cpu')
    gpu_t,gpu_y,gpu_g,gpu_state=timed(gpu,x.cuda(),'cuda')
    torch.testing.assert_close(cpu_y,gpu_y,rtol=2e-3,atol=1e-5)
    torch.testing.assert_close(cpu_g,gpu_g,rtol=2e-3,atol=1e-5)
    timestep=[]
    small=x[:4,:32]
    for steps in (8,16,32,64):
        r=YIGReservoir(10,replace(c,substeps=steps),7)
        with torch.no_grad(): values,state=r(small)
        timestep.append((steps,values,state[0]))
    reference=timestep[-1][2]
    convergence={str(s):float((z-reference).norm()/reference.norm()) for s,y,z in timestep[:-1]}
    record=dict(gpu_name=torch.cuda.get_device_name(0),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        torch_version=torch.__version__,torch_cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),cpu_seconds=cpu_t,gpu_seconds=gpu_t,
        speedup=cpu_t/gpu_t,feature_max_abs_error=float((cpu_y-gpu_y).abs().max()),
        gradient_max_abs_error=float((cpu_g-gpu_g).abs().max()),
        action_norm_relative_error=float(abs(cpu_state.abs().square().sum()-gpu_state.abs().square().sum())/cpu_state.abs().square().sum()),
        timestep_relative_state_error_vs_64=convergence,physics_config=c.__dict__,
        approximation_gate='must review timestep error before science; no automatic pass from CUDA visibility')
    (root/'results/physics-preflight.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2),flush=True)
