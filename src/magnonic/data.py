"""Disjoint character splits and independent synthetic delayed-recall seeds."""
import hashlib, json, pathlib
import numpy as np
import torch

def load_characters(path: pathlib.Path):
    raw=path.read_bytes()
    text=raw.decode('utf-8')
    cuts=(int(len(text)*.90),int(len(text)*.95))
    vocabulary=sorted(set(text[:cuts[0]]))
    mapping={c:i for i,c in enumerate(vocabulary)}
    parts=(text[:cuts[0]],text[cuts[0]:cuts[1]],text[cuts[1]:])
    unknown=set(''.join(parts))-set(vocabulary)
    if unknown:
        raise ValueError(f'Characters outside training vocabulary: {unknown}')
    arrays={name:torch.tensor([mapping[c] for c in part],dtype=torch.long)
            for name,part in zip(('train','validation','test'),parts)}
    metadata=dict(sha256=hashlib.sha256(raw).hexdigest(),vocabulary=vocabulary,
        split_offsets=[0,*cuts,len(text)],split_lengths={k:len(v) for k,v in arrays.items()},
        tokenizer='training-only character vocabulary',boundary='state reset at every sampled window')
    return arrays,metadata

def character_batch(array: torch.Tensor,batch: int,length: int,generator: torch.Generator,
                    device: str):
    starts=torch.randint(len(array)-length,(batch,),generator=generator)
    indices=starts[:,None]+torch.arange(length+1)[None,:]
    windows=array[indices].to(device)
    return windows[:,:-1],windows[:,1:]

def recall_batch(batch: int,delay: int,seed: int,device: str):
    """One of eight symbols, delay blank inputs, then a query. Last target is symbol.

    No label/token reaches the readout. Evaluation seeds never train.
    This is delayed recall, not selective copying or natural language.
    """
    rng=np.random.default_rng(seed)
    symbol=rng.integers(0,8,size=batch)
    tokens=np.full((batch,delay+2),8,dtype=np.int64)
    tokens[:,0]=symbol
    tokens[:,-1]=9
    return torch.tensor(tokens,device=device),torch.tensor(symbol,device=device)
