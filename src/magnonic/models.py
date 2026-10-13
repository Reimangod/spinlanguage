"""Same linear readout for all physical variants; a separate digital GRU comparator."""
import torch
from torch import nn
from .physics import YIGReservoir
from .config import PhysicsConfig

class PhysicalLanguageModel(nn.Module):
    def __init__(self, vocabulary: int, config: PhysicsConfig, seed: int, variant: str):
        super().__init__()
        self.reservoir=YIGReservoir(vocabulary,config,seed,variant=='trained',variant=='no_memory')
        self.readout=nn.Linear(self.reservoir.features,vocabulary)

    def forward(self,tokens,state=None):
        features,state=self.reservoir(tokens,state)
        return self.readout(features),state

class TinyGRU(nn.Module):
    def __init__(self,vocabulary: int,hidden: int=8):
        super().__init__()
        self.vocabulary=vocabulary
        self.gru=nn.GRU(vocabulary,hidden,batch_first=True)
        self.readout=nn.Linear(hidden,vocabulary)

    def forward(self,tokens,state=None):
        inputs=torch.nn.functional.one_hot(tokens,self.vocabulary).float()
        values,state=self.gru(inputs,state)
        return self.readout(values),state
