"""CPU replay of saved checkpoints: mode traces and additional feasibility probes."""
import argparse,json,math,pathlib
from dataclasses import replace
import numpy as np
import torch
from magnonic.config import PhysicsConfig
from magnonic.models import PhysicalLanguageModel
from magnonic.data import load_characters

ROOT=pathlib.Path(__file__).resolve().parents[1]

@torch.no_grad()
def ce_cpu(model,array,length=32,windows=128):
    windows=min(windows,(len(array)-1)//length)
    starts=torch.arange(windows)*length
    values=array[starts[:,None]+torch.arange(length+1)]
    logits,_=model(values[:,:-1])
    return float(torch.nn.functional.cross_entropy(logits.flatten(0,1),values[:,1:].flatten()))

@torch.no_grad()
def replay(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    config=PhysicsConfig(**checkpoint['physics'])
    vocab=checkpoint['model']['readout.bias'].shape[0]
    model=PhysicalLanguageModel(vocab,config,checkpoint['seed'],'trained')
    model.load_state_dict(checkpoint['model'])
    return model,config

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=pathlib.Path,required=True)
    study=parser.parse_args().study
    torch.set_num_threads(1)
    arrays,metadata=load_characters(ROOT/'data/tinyshakespeare.txt')
    output=ROOT/'results/aggregate';output.mkdir(exist_ok=True)
    records=[]
    for path in study.glob('worker-*/results/lm-seed-11-*/best.pt'):
        variant=path.parent.name.split('-')[-1]
        if variant not in ('fixed','trained'):continue
        model,config=replay(path)
        original_coils=model.reservoir.coil_field_T_A.clone()
        original_sensors=model.reservoir.sensor_profiles.clone()
        ref=ce_cpu(model,arrays['test'])
        conditions=[]
        for name in ('two_sensors','coil_gain_drift_plus_5pct','coil_gain_drift_minus_5pct','extra_coil_crosstalk_10pct','half_timestep'):
            model.reservoir.coil_field_T_A.copy_(original_coils)
            model.reservoir.sensor_profiles.copy_(original_sensors)
            model.reservoir.config=config
            if name=='two_sensors':model.reservoir.sensor_profiles[2:]=0
            if name=='coil_gain_drift_plus_5pct':model.reservoir.coil_field_T_A.mul_(1.05)
            if name=='coil_gain_drift_minus_5pct':model.reservoir.coil_field_T_A.mul_(.95)
            if name=='extra_coil_crosstalk_10pct':
                model.reservoir.coil_field_T_A.copy_(.9*original_coils+.1*original_coils.roll(1,dims=1))
            if name=='half_timestep':
                model.reservoir.config=replace(config,substeps=config.substeps*2)
                dt=model.reservoir.config.token_duration_s/model.reservoir.config.substeps
                omega=model.reservoir.omega_rad_s
                model.reservoir.propagator_half=torch.exp((-config.gilbert_alpha*omega-1j*(omega-omega[0]))*dt/2).to(torch.complex64)
            ce=ce_cpu(model,arrays['test'])
            conditions.append(dict(condition=name,ce_nats=ce,delta_ce=ce-ref))
        records.append(dict(variant=variant,seed=11,reference_ce=ref,conditions=conditions,backend='PyTorch CPU',
            context='128 fixed test windows; evaluation only, not model selection; sensor removal has no readout refit'))
        model,config=replay(path)
        state=model.reservoir.zero_state(1)
        field=model.reservoir.local_field_T()
        mode=[];observation=[]
        tokens=arrays['validation'][:64]
        for token in tokens:
            observed,state=model.reservoir.step_token(token.reshape(1),state,field)
            mode.append(state[0].detach().numpy()[0]);observation.append(observed.detach().numpy()[0])
        np.savez(output/f'waveforms-{variant}.npz',mode=np.asarray(mode),observation=np.asarray(observation),
            token_duration_s=config.token_duration_s,cell_spacing_m=config.cell_spacing_m,
            source_checkpoint=str(path),input_tokens=tokens.numpy())
    (output/'additional_feasibility.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(records,indent=2))
