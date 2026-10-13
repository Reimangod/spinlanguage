"""Fixed-budget BPTT, validation-only checkpoint selection, paired seeds."""
import base64,csv,io,json,math,pathlib,time
from dataclasses import asdict,replace
import numpy as np
import torch
from torch.nn import functional as F
from .config import PhysicsConfig,TrainConfig
from .models import PhysicalLanguageModel,TinyGRU
from .data import character_batch,recall_batch

def make_model(vocabulary:int,physics:PhysicsConfig,seed:int,variant:str):
    torch.manual_seed(seed)
    model=TinyGRU(vocabulary) if variant=='gru' else PhysicalLanguageModel(vocabulary,physics,seed,variant)
    return model.cuda()

def checkpoint_stream(path:pathlib.Path):
    """Durable Mac copy after each finished run, in addition to remote checkpoint."""
    print('@@CHECKPOINT '+path.name+' '+base64.b64encode(path.read_bytes()).decode(),flush=True)

@torch.no_grad()
def evaluate_characters(model,array:torch.Tensor,length:int,batch:int=512,limit:int=0):
    model.eval()
    windows=(len(array)-1)//length
    if limit: windows=min(limit,windows)
    loss_sum=0.; correct=0; count=0
    for first in range(0,windows,batch):
        starts=torch.arange(first,min(first+batch,windows))*length
        indices=starts[:,None]+torch.arange(length+1)
        data=array[indices].cuda()
        logits,_=model(data[:,:-1])
        target=data[:,1:]
        loss_sum+=float(F.cross_entropy(logits.flatten(0,1),target.flatten(),reduction='sum'))
        correct+=int((logits.argmax(-1)==target).sum())
        count+=target.numel()
    return dict(ce_nats=loss_sum/count,bpc=loss_sum/count/math.log(2),accuracy=correct/count,tokens=count,
                protocol='nonoverlapping windows; causal next character; state resets at window start')

@torch.no_grad()
def evaluate_memory(model,delay:int,seed:int,batch:int=4096):
    model.eval()
    tokens,target=recall_batch(batch,delay,seed,'cuda')
    logits,_=model(tokens)
    predicted=logits[:,-1]
    return dict(delay=delay,ce_nats=float(F.cross_entropy(predicted,target)),
        accuracy=float((predicted.argmax(-1)==target).float().mean()),examples=batch,seed=seed)

def optimize(model,task:str,seed:int,physics:PhysicsConfig,train:TrainConfig,
             updates:int,output:pathlib.Path,arrays=None):
    output.mkdir(parents=True,exist_ok=False)
    groups=[dict(params=model.readout.parameters(),lr=train.readout_lr)]
    if isinstance(model,TinyGRU):
        groups.append(dict(params=model.gru.parameters(),lr=.003))
    elif model.reservoir.raw_current.requires_grad:
        groups.append(dict(params=[model.reservoir.raw_current],lr=train.current_lr))
    optimizer=torch.optim.Adam(groups)
    generator=torch.Generator().manual_seed(seed+20000)
    best=math.inf; best_step=0; best_state=None
    start=time.perf_counter()
    curves=[]
    torch.cuda.reset_peak_memory_stats()
    for update in range(1,updates+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        if task=='lm':
            tokens,targets=character_batch(arrays['train'],train.batch_size,train.sequence_length,generator,'cuda')
            logits,state=model(tokens)
            loss=F.cross_entropy(logits.flatten(0,1),targets.flatten())
        else:
            delay=train.memory_delays[(update-1)%len(train.memory_delays)]
            tokens,targets=recall_batch(train.batch_size,delay,seed*100000+update,'cuda')
            logits,state=model(tokens)
            loss=F.cross_entropy(logits[:,-1],targets)
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
        loss.backward()
        grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),train.gradient_clip,error_if_nonfinite=True))
        optimizer.step()
        if isinstance(model,PhysicalLanguageModel):
            maximum=float(state[0].detach().abs().max())
            # Keep reduced amplitudes inside a conservative small-angle domain.
            if maximum>.20: raise FloatingPointError(f'Envelope amplitude {maximum} exceeds reduction gate 0.20')
        else: maximum=None
        if update%train.validation_every==0 or update==updates or update==1:
            if task=='lm':
                validation=evaluate_characters(model,arrays['validation'],train.sequence_length,limit=train.validation_windows)
                value=validation['ce_nats']
            else:
                validation=[evaluate_memory(model,d,5000000+seed*100+d,batch=1024) for d in train.memory_delays]
                value=float(np.mean([v['ce_nats'] for v in validation]))
            row=dict(update=update,train_loss=float(loss.detach()),validation_ce=value,
                elapsed_seconds=time.perf_counter()-start,gradient_norm=grad_norm,max_amplitude=maximum)
            curves.append(row)
            print(json.dumps(dict(event='training',run=output.name,**row)),flush=True)
            if value<best:
                best=value; best_step=update
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),update=update,
                generator_state=generator.get_state(),torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),
                best_state=best_state,best_validation_ce=best,best_update=best_step,
                physics=asdict(physics),train=asdict(train),seed=seed,task=task)
            torch.save(checkpoint,output/'last.pt')
            with (output/'curve.csv').open('w') as file:
                writer=csv.DictWriter(file,fieldnames=list(row)); writer.writeheader(); writer.writerows(curves)
    model.load_state_dict(best_state)
    torch.save(dict(model=best_state,physics=asdict(physics),train=asdict(train),seed=seed,task=task,
                    best_update=best_step,best_validation_ce=best),output/'best.pt')
    summary=dict(updates=updates,best_update=best_step,best_validation_ce=best,
        training_seconds=time.perf_counter()-start,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
        parameters_total=sum(p.numel() for p in model.parameters()),
        parameters_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        readout_parameters=sum(p.numel() for p in model.readout.parameters()),
        training_tokens=updates*train.batch_size*train.sequence_length if task=='lm' else None,
        backend='PyTorch CUDA float32 / complex64',reset='each independent training window/recall example',
        bptt_cut='window length; no history carry between randomly sampled windows')
    if isinstance(model,PhysicalLanguageModel):
        summary.update(currents_A=model.reservoir.currents_A().detach().cpu().tolist(),
            local_field_T=model.reservoir.local_field_T().detach().cpu().tolist(),
            linear_readout_multiply_adds_per_character=model.reservoir.features*model.readout.out_features,
            features=model.reservoir.features)
    (output/'training.json').write_text(json.dumps(summary,indent=2))
    # State files are tiny. Stream a completed-run snapshot to the control Mac.
    return summary

@torch.no_grad()
def generate(model,vocabulary:list[str],characters:int,temperature:float,seed:int):
    model.eval(); torch.manual_seed(seed)
    device=next(model.parameters()).device
    mapping={c:i for i,c in enumerate(vocabulary)}
    prompt='First Citizen:\n'
    state=None
    for char in prompt:
        logits,state=model(torch.tensor([[mapping[char]]],device=device),state)
    generated=prompt
    for _ in range(characters):
        index=int(torch.multinomial(torch.softmax(logits[0,-1]/temperature,dim=-1),1))
        generated+=vocabulary[index]
        logits,state=model(torch.tensor([[index]],device=device),state)
    return generated

@torch.no_grad()
def experimental_ablations(model,array,physics,train,output):
    """Post-training evaluation only; no tuning or selection on test values."""
    if not isinstance(model,PhysicalLanguageModel): return []
    variants=[('reference',{}),('sensor_noise_001',dict(sensor_noise_std=.01)),
        ('sensor_noise_005',dict(sensor_noise_std=.05)),('dac_4bit',dict(dac_bits=4)),
        ('dac_8bit',dict(dac_bits=8)),('sensor_bandwidth_25MHz',dict(sensor_bandwidth_Hz=25e6)),
        ('thermal_300K',dict(thermal_noise=True)),('linear_wave',dict(nonlinear_scale=0)),
        ('current_limit_half',dict(current_limit_A=.2))]
    records=[]
    for name,changes in variants:
        model.reservoir.config=replace(physics,**changes)
        torch.manual_seed(80001)
        result=evaluate_characters(model,array,train.sequence_length,limit=128)
        records.append(dict(condition=name,changes=changes,**result))
    model.reservoir.config=physics
    (output/'ablations.json').write_text(json.dumps(records,indent=2))
    return records
