"""Generate characters from an already trained, trusted local checkpoint."""
import argparse,pathlib
import torch
from magnonic.config import PhysicsConfig
from magnonic.data import load_characters
from magnonic.models import PhysicalLanguageModel,TinyGRU
from magnonic.training import generate
ROOT=pathlib.Path(__file__).resolve().parents[1]
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=pathlib.Path,required=True)
    parser.add_argument('--variant',choices=['fixed','trained','no_memory','gru'],required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--characters',type=int,default=400)
    parser.add_argument('--seed',type=int,default=123)
    args=parser.parse_args()
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    _,metadata=load_characters(ROOT/'data/tinyshakespeare.txt')
    vocab=len(metadata['vocabulary'])
    model=TinyGRU(vocab) if args.variant=='gru' else PhysicalLanguageModel(vocab,PhysicsConfig(**checkpoint['physics']),checkpoint['seed'],args.variant)
    model.load_state_dict(checkpoint['model']);model.to(args.device)
    print(generate(model,metadata['vocabulary'],args.characters,.8,args.seed))
