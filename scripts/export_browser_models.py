"""Export actual best.pt weights plus PyTorch parity fixtures for browser inference.

No retraining or synthetic replacement weights. Metadata records source SHA-256.
Complex buffers are exported as real/imaginary pairs. The web runtime implements
the same deterministic, noise-free forward equations on the visitor's CPU.
"""
from __future__ import annotations
import hashlib,json,pathlib
import torch
from magnonic.config import PhysicsConfig
from magnonic.models import PhysicalLanguageModel,TinyGRU

ROOT=pathlib.Path(__file__).resolve().parents[1]
STUDY=ROOT/'results/study-20261014-061533'

def tensor_json(value:torch.Tensor):
    return (torch.view_as_real(value) if value.is_complex() else value).tolist()

if __name__=='__main__':
    torch.set_num_threads(1)
    identity=json.loads((STUDY/'worker-0/results/identity.json').read_text())
    vocabulary=identity['dataset']['vocabulary']
    mapping={char:i for i,char in enumerate(vocabulary)}
    records=[];fixtures=[]
    variants=('trained','fixed','no_memory','gru')
    prompts=('First Citizen:\nBefore we proceed, hear me speak.',
             'ROMEO:\nWhat light through yonder window breaks?',
             'To be, or not to be: that is the question.\n'*3)
    for seed in (11,22,33):
        for variant in variants:
            paths=list(STUDY.glob(f'worker-*/results/lm-seed-{seed}-{variant}/best.pt'))
            if len(paths)!=1:raise ValueError('Expected one original checkpoint')
            path=paths[0]
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            config=PhysicsConfig(**checkpoint['physics'])
            model=TinyGRU(len(vocabulary)) if variant=='gru' else PhysicalLanguageModel(len(vocabulary),config,seed,variant)
            model.load_state_dict(checkpoint['model']);model.eval()
            summary=json.loads((path.parent/'summary.json').read_text())
            record=dict(id=f'{variant}-{seed}',variant=variant,seed=seed,physics=checkpoint['physics'],
                weights={k:tensor_json(v) for k,v in model.state_dict().items() if k!='reservoir.omega_rad_s'},
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                checkpoint_path=str(path.relative_to(ROOT)),best_update=checkpoint['best_update'],
                evaluation=summary['evaluation'],parameters=summary['parameters_trainable'])
            records.append(record)
            with torch.inference_mode():
                for prompt in prompts:
                    tokens=torch.tensor([[mapping[c] for c in prompt]])
                    logits,state=model(tokens)
                    indices=(0,5,len(prompt)-1)
                    fixtures.append(dict(model_id=record['id'],prompt=prompt,
                        logits=[dict(position=i,values=logits[0,i].tolist()) for i in indices],
                        state=[tensor_json(s[0]) for s in state] if variant!='gru' else tensor_json(state[0,0])))
    output=ROOT/'ui/models.json'
    output.write_text(json.dumps(dict(format_version=1,vocabulary=vocabulary,models=records,
        provenance='Original A100-trained best checkpoints; deterministic forward, browser CPU',
        aggregate=json.loads((ROOT/'results/aggregate/metrics.json').read_text())['language']),separators=(',',':')))
    (ROOT/'tests/browser_parity_fixtures.json').write_text(json.dumps(fixtures,separators=(',',':')))
    print(json.dumps(dict(models=len(records),model_bytes=output.stat().st_size,parity_cases=len(fixtures))))
