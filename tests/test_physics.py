"""Invariants and finite differences for the chosen reduction, not full LLG proof."""
import math
from dataclasses import replace
import torch
from magnonic.config import PhysicsConfig
from magnonic.physics import YIGReservoir,dispersion_rad_s,MU0

def precise(config=PhysicsConfig(),seed=7):
    r=YIGReservoir(10,config,seed).double()
    r.propagator_half=r.propagator_half.to(torch.complex128)
    r.token_code=r.token_code.to(torch.complex128)
    return r

def test_uniform_frequency_and_units():
    c=PhysicsConfig()
    frequency=dispersion_rad_s(torch.zeros(1,dtype=torch.float64),c).item()
    assert math.isclose(frequency,c.gyromagnetic_rad_s_T*(c.bias_T-MU0*c.saturation_magnetization_A_m),rel_tol=1e-12)
    assert 4e9<frequency/(2*math.pi)<6e9

def test_linear_mode_exact_decay_and_phase():
    c=replace(PhysicsConfig(),nonlinear_scale=0)
    r=precise(c)
    amplitude=torch.zeros(1,c.cells,dtype=torch.complex128)
    amplitude[:]=0.01
    out=r.wave_step(amplitude,torch.zeros_like(amplitude),torch.zeros(c.cells,dtype=torch.float64))
    expected=math.exp(-c.gilbert_alpha*r.carrier_rad_s*c.token_duration_s/c.substeps)
    torch.testing.assert_close(out.abs(),amplitude.abs()*expected,rtol=2e-7,atol=1e-10)

def test_undriven_norm_and_propagation():
    c=replace(PhysicsConfig(),gilbert_alpha=0)
    r=precise(c)
    a=torch.zeros(1,c.cells,dtype=torch.complex128); a[0,0]=.05
    norm=a.abs().square().sum()
    b=r.wave_step(a,torch.zeros_like(a),r.local_field_T())
    torch.testing.assert_close(b.abs().square().sum(),norm,rtol=2e-7,atol=1e-10)
    assert b[0,1:].abs().max()>1e-4

def test_coil_bounds():
    r=precise()
    with torch.no_grad(): r.raw_current.fill_(100)
    assert r.currents_A().abs().max()<=r.config.current_limit_A
    assert r.local_field_T().abs().max()<=.0020000001

def test_current_gradient_finite_difference():
    r=precise()
    x=torch.tensor([[1,2,3,4,8,9]])
    out,_=r(x); loss=out.square().mean(); loss.backward()
    analytical=r.raw_current.grad[0].item()
    with torch.no_grad():
        orig=r.raw_current[0].item(); epsilon=1e-4
        r.raw_current[0]=orig+epsilon; plus=r(x)[0].square().mean().item()
        r.raw_current[0]=orig-epsilon; minus=r(x)[0].square().mean().item()
        r.raw_current[0]=orig
    numerical=(plus-minus)/(2*epsilon)
    assert abs(analytical)>1e-8
    assert math.isclose(analytical,numerical,rel_tol=2e-4,abs_tol=1e-7)

def test_causality_and_state_reset():
    r=precise()
    first=torch.tensor([[1,2,3,4]])
    second=torch.tensor([[1,2,8,9]])
    a,_=r(first); b,_=r(second)
    torch.testing.assert_close(a[:,:2],b[:,:2])
    torch.testing.assert_close(a,r(first)[0])
    left,state=r(first[:,:2]); right,_=r(first[:,2:],state)
    torch.testing.assert_close(torch.cat((left,right),dim=1),a)

def test_no_memory_and_bptt_cut():
    r=precise(); r.memory_disabled=True
    a,_=r(torch.tensor([[1,9]])); b,_=r(torch.tensor([[2,9]]))
    torch.testing.assert_close(a[:,-1],b[:,-1])
    r.memory_disabled=False
    _,state=r(torch.tensor([[1,2]]))
    assert state[0].grad_fn is not None
    detached=r.detach_state(state)
    assert all(s.grad_fn is None for s in detached)

def test_quantization_levels_and_noise_reproducibility():
    r=precise(replace(PhysicsConfig(),dac_bits=4,thermal_noise=True))
    units=(r.currents_A()/r.config.current_limit_A+1)*15/2
    torch.testing.assert_close(units,units.round(),rtol=0,atol=1e-12)
    x=torch.tensor([[1,2,3]])
    torch.manual_seed(123); a=r(x)[0]
    torch.manual_seed(123); b=r(x)[0]
    torch.testing.assert_close(a,b)
