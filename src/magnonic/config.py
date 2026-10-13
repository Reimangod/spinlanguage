"""Explicit SI parameters. See docs/THEORY_AUDIT.md for provenance."""
from dataclasses import dataclass

@dataclass(frozen=True)
class PhysicsConfig:
    cells: int = 32
    cell_spacing_m: float = 25e-6
    film_thickness_m: float = 1e-6
    waveguide_width_m: float = 50e-6
    saturation_magnetization_A_m: float = 140000.0
    exchange_J_m: float = 3.5e-12
    gyromagnetic_rad_s_T: float = 1.760859e11
    bias_T: float = 0.350
    gilbert_alpha: float = 2e-4
    token_duration_s: float = 8e-9
    substeps: int = 8
    control_coils: int = 8
    current_limit_A: float = 0.4
    field_gain_T_A: float = 0.005
    coil_width_m: float = 75e-6
    dac_bits: int = 0
    input_ports: int = 4
    input_field_limit_T: float = 5e-6
    input_bandwidth_Hz: float = 200e6
    sensor_ports: int = 4
    sensor_width_m: float = 25e-6
    sensor_bandwidth_Hz: float = 100e6
    samples_per_token: int = 2
    sensor_scale: float = 100.0
    sensor_noise_std: float = 0.0
    temperature_K: float = 300.0
    thermal_noise: bool = False
    nonlinear_scale: float = 1.0
    port_offset_cells: int = 0

@dataclass(frozen=True)
class TrainConfig:
    seeds: tuple[int,...] = (11, 22, 33)
    batch_size: int = 512
    sequence_length: int = 32
    lm_updates: int = 600
    memory_updates: int = 300
    search_updates: int = 150
    validation_every: int = 50
    readout_lr: float = 0.015
    current_lr: float = 0.025
    gradient_clip: float = 1.0
    validation_windows: int = 0
    test_windows: int = 0
    memory_delays: tuple[int,...] = (1,4,8,16,24)
    generation_characters: int = 400
    generation_temperature: float = 0.8
