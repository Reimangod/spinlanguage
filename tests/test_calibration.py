import json,torch,pytest
from magnonic.config import PhysicsConfig
from magnonic.physics import YIGReservoir
from magnonic.calibration import apply_calibration

def test_calibration_units_provenance_and_field_path(tmp_path):
    model=YIGReservoir(10,PhysicsConfig(),11)
    path=tmp_path/'calibration.json'
    record=dict(provenance='simulated_placeholder',coil_units='T/A',coil_field_T_A=model.coil_field_T_A.tolist())
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):apply_calibration(model,path)
    record['provenance']='field_solver';record['source']='test-only field solver fixture, not measured'
    record['coil_units']='mT/A';path.write_text(json.dumps(record))
    with pytest.raises(ValueError):apply_calibration(model,path)
    record['coil_units']='T/A'
    record['coil_field_T_A']=(model.coil_field_T_A*.5).tolist()
    before=model.local_field_T().detach().clone()
    path.write_text(json.dumps(record));apply_calibration(model,path)
    torch.testing.assert_close(model.local_field_T(),before*.5)
