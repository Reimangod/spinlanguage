"""Uniform-thickness FVMSW Galerkin model with bounded DC field control.

This implements a proposed rotating-wave reduction, not SpinTorch or a full
micromagnetic reproduction. A periodic Fourier basis retains propagation.
"""
import math
from dataclasses import replace
import torch
from torch import nn
from .config import PhysicsConfig

MU0=4*math.pi*1e-7
KB_J_K=1.380649e-23

def dispersion_rad_s(k_rad_m: torch.Tensor, config: PhysicsConfig) -> torch.Tensor:
    """Lowest uniform-thickness, perpendicularly saturated dipole-exchange mode.

    h = B0-mu0 Ms + 2 A k^2/Ms (tesla), P=1-(1-exp(-|k|d))/(|k|d).
    Diagonal approximation omits coupling to perpendicular standing modes.
    """
    q=k_rad_m.abs()*config.film_thickness_m
    safe_q=q.clamp_min(1e-12)
    p=torch.where(q<1e-7,q/2-q*q/6,1+torch.expm1(-q)/safe_q)
    h=config.bias_T-MU0*config.saturation_magnetization_A_m
    h=h+2*config.exchange_J_m*k_rad_m.square()/config.saturation_magnetization_A_m
    return config.gyromagnetic_rad_s_T*torch.sqrt(h*(h+MU0*config.saturation_magnetization_A_m*p))

def gaussian_ports(positions: torch.Tensor, centers: torch.Tensor, width_m: float,
                   length_m: float) -> torch.Tensor:
    """Periodic Gaussian antenna averages; each row integrates to one."""
    distance=(positions[None,:]-centers[:,None]+length_m/2)%length_m-length_m/2
    profiles=torch.exp(-0.5*(distance/width_m).square())
    return profiles/profiles.sum(dim=1,keepdim=True)

class YIGReservoir(nn.Module):
    def __init__(self, vocabulary: int, config: PhysicsConfig, seed: int,
                 train_internal: bool=True, memory_disabled: bool=False):
        super().__init__()
        self.config=config
        self.memory_disabled=memory_disabled
        n=config.cells
        length=n*config.cell_spacing_m
        x=torch.arange(n,dtype=torch.float64)*config.cell_spacing_m
        k=2*math.pi*torch.fft.fftfreq(n,d=config.cell_spacing_m,dtype=torch.float64)
        omega=dispersion_rad_s(k,config)
        # RF reference uses k=0. Every retained mode is represented explicitly.
        carrier=omega[0]
        dt=config.token_duration_s/config.substeps
        half_step=torch.exp((-config.gilbert_alpha*omega-1j*(omega-carrier))*dt/2)
        self.register_buffer('propagator_half',half_step.to(torch.complex64))
        self.register_buffer('omega_rad_s',omega)
        centers=torch.arange(config.control_coils,dtype=torch.float64)*length/config.control_coils
        coil=gaussian_ports(x,centers,config.coil_width_m,length)
        # Peak response per coil; normalized row sum prevents overlapping coils
        # exceeding the stated +/-2 mT field envelope under saturated currents.
        coil=coil/coil.max(dim=1,keepdim=True).values
        coil=coil/coil.sum(dim=0,keepdim=True).clamp_min(1)
        self.register_buffer('coil_field_T_A',(coil.T*config.field_gain_T_A).float())
        offset=config.port_offset_cells*config.cell_spacing_m
        inputs=(torch.arange(config.input_ports,dtype=torch.float64)*length/config.input_ports+offset)%length
        sensors=(torch.arange(config.sensor_ports,dtype=torch.float64)*length/config.sensor_ports+length/8)%length
        self.register_buffer('input_profiles',gaussian_ports(x,inputs,config.sensor_width_m,length).float())
        self.register_buffer('sensor_profiles',gaussian_ports(x,sensors,config.sensor_width_m,length).float())
        gen=torch.Generator().manual_seed(seed+1234)
        code=torch.randn(vocabulary,config.input_ports,2,generator=gen)
        code=code/code.abs().amax(dim=(1,2),keepdim=True)
        self.register_buffer('token_code',torch.complex(code[...,0],code[...,1])/math.sqrt(2))
        self.raw_current=nn.Parameter(torch.randn(config.control_coils,generator=gen)*0.35,
                                      requires_grad=train_internal)
        self.carrier_rad_s=float(carrier)

    @property
    def features(self) -> int:
        return self.config.sensor_ports*self.config.samples_per_token*2

    def currents_A(self) -> torch.Tensor:
        current=self.config.current_limit_A*torch.tanh(self.raw_current)
        if self.config.dac_bits:
            levels=2**self.config.dac_bits-1
            rounded=torch.round((current/self.config.current_limit_A+1)*levels/2)*2/levels-1
            # Evaluation quantization is real; straight-through only if trained.
            discrete=rounded*self.config.current_limit_A
            current=current+(discrete-current).detach()
        return current

    def local_field_T(self) -> torch.Tensor:
        return self.coil_field_T_A@self.currents_A()

    def zero_state(self, batch: int) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        device=self.raw_current.device
        dtype=torch.complex128 if self.raw_current.dtype==torch.float64 else torch.complex64
        return (torch.zeros(batch,self.config.cells,device=device,dtype=dtype),
                torch.zeros(batch,self.config.input_ports,device=device,dtype=dtype),
                torch.zeros(batch,self.config.sensor_ports,device=device,dtype=dtype))

    def wave_step(self, amplitude: torch.Tensor, drive: torch.Tensor,
                  field_T: torch.Tensor) -> torch.Tensor:
        """Symmetric dispersion / drive / local Kerr split, exact subflows.

        Kerr coefficient gamma*mu0*Ms/2 is a circular-precession expansion of
        local perpendicular demagnetization. Nonlocal quartic terms omitted.
        """
        c=self.config
        dt=c.token_duration_s/c.substeps
        amplitude=torch.fft.ifft(torch.fft.fft(amplitude)*self.propagator_half)
        amplitude=amplitude+dt*drive/2
        nu=c.nonlinear_scale*c.gyromagnetic_rad_s_T*MU0*c.saturation_magnetization_A_m/2
        phase=dt*(c.gyromagnetic_rad_s_T*field_T+nu*amplitude.abs().square())
        amplitude=amplitude*torch.exp(-1j*phase)
        amplitude=amplitude+dt*drive/2
        amplitude=torch.fft.ifft(torch.fft.fft(amplitude)*self.propagator_half)
        if c.thermal_noise:
            volume=c.cell_spacing_m*c.waveguide_width_m*c.film_thickness_m
            b=c.bias_T-MU0*c.saturation_magnetization_A_m
            sigma=math.sqrt((1-math.exp(-2*c.gilbert_alpha*self.carrier_rad_s*dt))*KB_J_K*c.temperature_K/
                            (c.saturation_magnetization_A_m*volume*b))
            amplitude=amplitude+sigma*torch.complex(torch.randn_like(amplitude.real),torch.randn_like(amplitude.real))
        return amplitude

    def step_token(self, tokens: torch.Tensor,
                   state: tuple[torch.Tensor,torch.Tensor,torch.Tensor],
                   field_T: torch.Tensor) -> tuple[torch.Tensor,tuple[torch.Tensor,...]]:
        """Causal RF encoding, finite-bandwidth receiver and I/Q observations."""
        c=self.config
        amplitude,rf,sensor=state
        if self.memory_disabled:
            amplitude,rf,sensor=self.zero_state(tokens.shape[0])
        dt=c.token_duration_s/c.substeps
        input_half_decay=math.exp(-2*math.pi*c.input_bandwidth_Hz*dt/2)
        sensor_decay=math.exp(-2*math.pi*c.sensor_bandwidth_Hz*dt)
        desired=self.token_code[tokens]
        observations=[]
        for substep in range(c.substeps):
            rf=input_half_decay*rf+(1-input_half_decay)*desired
            drive=(rf@self.input_profiles.to(rf.dtype))*c.gyromagnetic_rad_s_T*c.input_field_limit_T/2
            before=amplitude
            amplitude=self.wave_step(amplitude,drive,field_T)
            rf=input_half_decay*rf+(1-input_half_decay)*desired
            observed=((before+amplitude)/2)@self.sensor_profiles.T.to(amplitude.dtype)
            sensor=sensor_decay*sensor+(1-sensor_decay)*observed
            if (substep+1)%(c.substeps//c.samples_per_token)==0:
                feature=torch.view_as_real(sensor).flatten(start_dim=1)*c.sensor_scale
                if c.sensor_noise_std:
                    feature=feature+c.sensor_noise_std*torch.randn_like(feature)
                observations.append(feature)
        return torch.cat(observations,dim=1),(amplitude,rf,sensor)

    def forward(self, tokens: torch.Tensor, state=None):
        """Returns current observations only; no token/history skip to readout."""
        if state is None:
            state=self.zero_state(tokens.shape[0])
        field=self.local_field_T()
        features=[]
        for column in tokens.unbind(dim=1):
            observation,state=self.step_token(column,state,field)
            features.append(observation)
        return torch.stack(features,dim=1),state

    def detach_state(self, state):
        """Explicit BPTT cut for inference/continuous streams; training resets windows."""
        return tuple(item.detach() for item in state)
