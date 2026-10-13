"""Load actual instrument calibration only when explicitly supplied; no fake data."""
import json,pathlib
import torch
from .physics import YIGReservoir

def apply_calibration(reservoir:YIGReservoir,path:pathlib.Path) -> dict:
    """Replace C[T/A] and antenna averages after validating units and provenance.

    The API is optional and not used in protocol v1; instrument gain, nonlinear
    coefficients and material parameters require a separate versioned protocol.
    """
    record=json.loads(path.read_text())
    if record.get('provenance') not in ('measured','field_solver'):
        raise ValueError('Calibration must declare measured or field_solver provenance')
    if record.get('coil_units')!='T/A': raise ValueError('coil_units must be T/A')
    matrix=torch.tensor(record['coil_field_T_A'],device=reservoir.raw_current.device,dtype=reservoir.raw_current.dtype)
    if matrix.shape!=reservoir.coil_field_T_A.shape or not torch.isfinite(matrix).all():
        raise ValueError('Invalid calibration matrix dimensions or values')
    reservoir.coil_field_T_A.copy_(matrix)
    return dict(provenance=record['provenance'],source=record.get('source'),applied_path=str(path))
