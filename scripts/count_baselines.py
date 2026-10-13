"""Diagnostic count baselines; distinct digital models, never physical readout inputs."""
import collections,json,math,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
text=(ROOT/'data/tinyshakespeare.txt').read_text()
train=text[:int(.9*len(text))]; test=text[int(.95*len(text)):]
vocab=sorted(set(train)); index={c:i for i,c in enumerate(vocab)}
counts=collections.Counter(train);pairs=collections.Counter(zip(train[:-1],train[1:]));previous=collections.Counter(train[:-1])
total=0;loss_uni=0.;loss_bi=0.;smoothing=.1
for start in range(0,((len(test)-1)//32)*32,32):
    window=test[start:start+33]
    for a,b in zip(window[:-1],window[1:]):
        loss_uni-=math.log((counts[b]+smoothing)/(len(train)+smoothing*len(vocab)))
        loss_bi-=math.log((pairs[(a,b)]+smoothing)/(previous[a]+smoothing*len(vocab)))
        total+=1
result=dict(unigram_ce=loss_uni/total,bigram_ce=loss_bi/total,tokens=total,
    unigram_bpc=loss_uni/total/math.log(2),bigram_bpc=loss_bi/total/math.log(2),
    smoothing=smoothing,backend='Python CPU counts',data='full training split; same nonoverlapping test windows',
    interpretation='diagnostic digital comparison with direct token access, not a physical reservoir')
(ROOT/'results/aggregate/count_baselines.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
