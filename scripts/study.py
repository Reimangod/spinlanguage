"""Disjoint two-worker allocation of a preregistered study, with validation search."""
import argparse,base64,io,json,os,pathlib,tarfile,time
from dataclasses import asdict,replace
import torch
from magnonic.config import PhysicsConfig,TrainConfig
from magnonic.data import load_characters
from magnonic.training import make_model,optimize,evaluate_characters,evaluate_memory,generate,experimental_ablations

ROOT=pathlib.Path(__file__).resolve().parents[1]
SHARED=pathlib.Path('/dev/shm/rei-magnonic-lm-20261014-v1')/('selection-'+os.environ['MAGNONIC_SOURCE_DIGEST'][:16])

def stream_results(directory):
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as tar:
        tar.add(directory,arcname='results/'+directory.name)
    print('@@ARTIFACT '+base64.b64encode(buffer.getvalue()).decode(),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--worker',type=int,choices=[0,1],required=True)
    worker=parser.parse_args().worker
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    physics=PhysicsConfig(); train=TrainConfig()
    arrays,metadata=load_characters(ROOT/'data/tinyshakespeare.txt')
    vocabulary=len(metadata['vocabulary'])
    identity=dict(worker=worker,job_id=os.environ.get('SLURM_JOB_ID'),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),gpu_name=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,torch_cuda=torch.version.cuda,logical_device=0,
        physics=asdict(physics),train=asdict(train),dataset=metadata)
    (ROOT/'results/identity.json').write_text(json.dumps(identity,indent=2))
    print(json.dumps(dict(event='identity',**identity)),flush=True)
    # Exactly two candidates, one per GPU, with the same inner budget and seed.
    SHARED.mkdir(exist_ok=True)
    candidate_offset=(0,4)[worker]
    candidate_physics=replace(physics,port_offset_cells=candidate_offset)
    model=make_model(vocabulary,candidate_physics,11,'trained')
    output=ROOT/'results'/f'search-offset-{candidate_offset}'
    summary=optimize(model,'lm',11,candidate_physics,train,train.search_updates,output,arrays)
    selection_record=dict(offset=candidate_offset,validation_ce=summary['best_validation_ce'],updates=train.search_updates,
                          worker=worker,test_used=False)
    (output/'selection.json').write_text(json.dumps(selection_record,indent=2))
    stream_results(output)
    temporary=SHARED/f'candidate-{worker}.tmp'
    temporary.write_text(json.dumps(selection_record)); temporary.rename(SHARED/f'candidate-{worker}.json')
    wait_start=time.monotonic()
    while not all((SHARED/f'candidate-{w}.json').exists() for w in (0,1)):
        if time.monotonic()-wait_start>1800: raise TimeoutError('paired candidate evidence did not arrive; do not rerun blindly')
        time.sleep(2)
    candidates=[json.loads((SHARED/f'candidate-{w}.json').read_text()) for w in (0,1)]
    selected=min(candidates,key=lambda x:(x['validation_ce'],x['offset']))
    physics=replace(physics,port_offset_cells=selected['offset'])
    (ROOT/'results/search-selection.json').write_text(json.dumps(dict(candidates=candidates,selected=selected),indent=2))
    print(json.dumps(dict(event='selected',**selected)),flush=True)
    # Six paired experiments, each allocated exactly once across the two GPUs.
    tasks=[('memory',s) for s in train.seeds]+[('lm',s) for s in train.seeds]
    for task_id,(task,seed) in enumerate(tasks):
        if task_id%2!=worker: continue
        variants=('fixed','trained','no_memory') if task=='memory' else ('fixed','trained','no_memory','gru')
        for variant in variants:
            model=make_model(10 if task=='memory' else vocabulary,physics,seed,variant)
            output=ROOT/'results'/f'{task}-seed-{seed}-{variant}'
            summary=optimize(model,task,seed,physics,train,train.memory_updates if task=='memory' else train.lm_updates,output,arrays)
            if task=='memory':
                evaluation=[evaluate_memory(model,d,9000000+seed*100+d) for d in train.memory_delays]
            else:
                evaluation=evaluate_characters(model,arrays['test'],train.sequence_length,limit=train.test_windows)
                text=generate(model,metadata['vocabulary'],train.generation_characters,train.generation_temperature,seed+99000)
                (output/'generation.txt').write_text(text)
                experimental_ablations(model,arrays['test'],physics,train,output)
            (output/'evaluation.json').write_text(json.dumps(evaluation,indent=2))
            (output/'summary.json').write_text(json.dumps(dict(task=task,seed=seed,variant=variant,**summary,evaluation=evaluation),indent=2))
            print(json.dumps(dict(event='completed',run=output.name,evaluation=evaluation)),flush=True)
            stream_results(output)
            del model; torch.cuda.empty_cache()
    (ROOT/'results/completed.json').write_text(json.dumps(dict(worker=worker,success=True)))
