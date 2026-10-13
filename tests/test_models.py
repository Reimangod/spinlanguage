"""Paired initialization and the absence of privileged readout inputs."""
import torch
from magnonic.config import PhysicsConfig
from magnonic.models import PhysicalLanguageModel

def test_fixed_and_trainable_pair_initialization():
    torch.manual_seed(11); fixed=PhysicalLanguageModel(65,PhysicsConfig(),11,'fixed')
    torch.manual_seed(11); trained=PhysicalLanguageModel(65,PhysicsConfig(),11,'trained')
    for name,value in fixed.state_dict().items():
        torch.testing.assert_close(value,trained.state_dict()[name])
    assert not fixed.reservoir.raw_current.requires_grad
    assert trained.reservoir.raw_current.requires_grad
    tokens=torch.tensor([[1,2,3,4]])
    torch.testing.assert_close(fixed(tokens)[0],trained(tokens)[0])

def test_readout_gets_only_sensor_features():
    model=PhysicalLanguageModel(65,PhysicsConfig(),11,'trained')
    observed=[]
    hook=model.readout.register_forward_pre_hook(lambda module,args: observed.append(args[0].detach().clone()))
    tokens=torch.tensor([[1,2,3]])
    logits,_=model(tokens)
    features,_=model.reservoir(tokens)
    hook.remove()
    assert observed[0].shape==(1,3,16)
    torch.testing.assert_close(observed[0],features)
    assert logits.shape==(1,3,65)
