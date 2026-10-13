"""Aggregate measured result files and create static research figures on CPU."""
import argparse,csv,json,pathlib,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
COLORS=dict(fixed='#3977b4',trained='#d15b3e',no_memory='#808080',gru='#45976b')

def save(fig,path):
    fig.savefig(path,dpi=180,bbox_inches='tight'); plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--study',type=pathlib.Path,required=True)
    study=parser.parse_args().study
    output=ROOT/'results/aggregate'; output.mkdir(exist_ok=True)
    summaries=[dict(path=str(p),**json.loads(p.read_text())) for p in study.glob('worker-*/results/*/summary.json')]
    (output/'runs.json').write_text(json.dumps(summaries,indent=2))
    main=[r for r in summaries if r['task']=='lm']
    memory=[r for r in summaries if r['task']=='memory']
    aggregate={}
    for variant in COLORS:
        records=[r for r in main if r['variant']==variant]
        if records:
            ce=[r['evaluation']['ce_nats'] for r in records]
            aggregate[variant]=dict(n=len(records),ce_mean=statistics.mean(ce),ce_sd=statistics.stdev(ce) if len(ce)>1 else 0,
                bpc_mean=statistics.mean(r['evaluation']['bpc'] for r in records),seeds=[r['seed'] for r in records],
                parameters_total=records[0]['parameters_total'],readout_parameters=records[0]['readout_parameters'])
    paired=[]
    for seed in (11,22,33):
        pair={r['variant']:r for r in main if r['seed']==seed}
        if 'fixed' in pair and 'trained' in pair:
            paired.append(dict(seed=seed,ce_trained_minus_fixed=pair['trained']['evaluation']['ce_nats']-pair['fixed']['evaluation']['ce_nats']))
    (output/'metrics.json').write_text(json.dumps(dict(language=aggregate,paired=paired,completion=dict(lm_runs=len(main),memory_runs=len(memory))),indent=2))
    with (output/'lm_metrics.csv').open('w') as file:
        fields=['seed','variant','ce_nats','bpc','accuracy','tokens','parameters_total','parameters_trainable','readout_parameters','training_seconds','peak_gpu_memory_bytes']
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for r in main:
            row={k:r.get(k) for k in fields};row.update(r['evaluation']);row.pop('protocol',None)
            writer.writerow(row)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for i,(variant,record) in enumerate(aggregate.items()):
        axes[0].bar(i,record['ce_mean'],yerr=record['ce_sd'],color=COLORS[variant],capsize=4)
    axes[0].set_xticks(range(len(aggregate)),list(aggregate));axes[0].set_ylabel('Test cross-entropy [nats]');axes[0].set_ylim(0,4)
    axes[0].set_title('Tiny Shakespeare: mean and SD across seeds')
    for r in main:
        rows=list(csv.DictReader(open(pathlib.Path(r['path']).parent/'curve.csv')))
        axes[1].plot([int(x['update']) for x in rows],[float(x['validation_ce']) for x in rows],color=COLORS[r['variant']],alpha=.55,label=r['variant'] if r['seed']==11 else None)
    axes[1].legend();axes[1].set_xlabel('Optimizer updates');axes[1].set_ylabel('Validation CE [nats]')
    save(fig,output/'language_comparison.png')
    fig,ax=plt.subplots(figsize=(7,4))
    delays=(1,4,8,16,24)
    for variant in ('fixed','trained','no_memory'):
        selected=[r for r in memory if r['variant']==variant]
        if not selected: continue
        matrix=np.array([[e['accuracy'] for e in r['evaluation']] for r in selected])
        ax.errorbar(delays,matrix.mean(0),yerr=matrix.std(0,ddof=1) if len(matrix)>1 else None,marker='o',label=variant,color=COLORS[variant],capsize=4)
    ax.axhline(1/8,color='black',ls='--',alpha=.5,label='chance 1/8');ax.set_ylim(0,1)
    ax.set_xlabel('Blank-symbol delay [tokens]');ax.set_ylabel('Recall test accuracy');ax.legend()
    ax.set_title('Delayed recall: calculated data, mean and SD across seeds')
    save(fig,output/'memory_delays.png')
    fig,ax=plt.subplots(figsize=(8,4))
    for variant in ('fixed','trained'):
        selected=[r for r in main if r['variant']==variant]
        if not selected: continue
        ablations=[json.loads((pathlib.Path(r['path']).parent/'ablations.json').read_text()) for r in selected]
        matrix=np.array([[a['ce_nats'] for a in items] for items in ablations])
        labels=[a['condition'] for a in ablations[0]]
        ax.plot(range(len(labels)),matrix.mean(0),marker='o',label=variant,color=COLORS[variant])
    if main:
        ax.set_xticks(range(len(labels)),labels,rotation=35,ha='right')
    ax.set_ylabel('Test-subset CE [nats]');ax.legend();ax.set_title('Post-training sensitivity; no retraining or model selection')
    save(fig,output/'experiment_ablations.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for variant in ('fixed','trained'):
        records=[r for r in main if r['variant']==variant]
        if not records: continue
        r=next((r for r in records if r['seed']==11),records[0])
        axes[0].plot(np.arange(len(r['currents_A'])),np.array(r['currents_A'])*1000,marker='o',label=variant,color=COLORS[variant])
        axes[1].plot(np.arange(len(r['local_field_T']))*25,np.array(r['local_field_T'])*1000,label=variant,color=COLORS[variant])
    axes[0].set_ylabel('DC current [mA]');axes[0].set_xlabel('Coil index');axes[0].legend()
    axes[1].set_ylabel('Local control field [mT]');axes[1].set_xlabel('Periodic coordinate [micrometre]');axes[1].legend()
    save(fig,output/'weight_distribution.png')
    print(json.dumps(dict(language=aggregate,paired=paired),indent=2))
