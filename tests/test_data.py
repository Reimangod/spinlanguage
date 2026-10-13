import pathlib,torch
from magnonic.data import load_characters,character_batch,recall_batch

def test_split_and_next_character(tmp_path):
    p=tmp_path/'text'; p.write_text('abcde'*100)
    arrays,meta=load_characters(p)
    assert meta['split_offsets']==[0,450,475,500]
    assert sum(len(v) for v in arrays.values())==500
    x,y=character_batch(arrays['train'],3,8,torch.Generator().manual_seed(2),'cpu')
    torch.testing.assert_close(x[:,1:],y[:,:-1])

def test_recall_rule_and_independent_split_seeds():
    a,y=recall_batch(128,8,123,'cpu')
    b,z=recall_batch(128,8,124,'cpu')
    torch.testing.assert_close(a[:,0],y)
    assert torch.all(a[:,1:-1]==8) and torch.all(a[:,-1]==9)
    assert not torch.equal(y,z)
