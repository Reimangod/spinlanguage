"""Completion audit against the preregistered scope, not merely file presence."""
import argparse,csv,hashlib,json,pathlib,statistics,tarfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=pathlib.Path,required=True)
    study=parser.parse_args().study
    checks=[]
    def require(name,condition,details=None):
        checks.append(dict(requirement=name,passed=bool(condition),details=details))
    protocol=json.loads((ROOT/'configs/protocol_v1.json').read_text())
    runs=[json.loads(p.read_text()) for p in study.glob('worker-*/results/*/summary.json')]
    expected={(task,s,v) for task,variants in [('memory',protocol['memory_variants']),('lm',protocol['lm_variants'])]
              for s in protocol['main_seeds'] for v in variants}
    observed=[(r['task'],r['seed'],r['variant']) for r in runs]
    require('Every main task/seed/variant executed once',len(observed)==len(expected) and set(observed)==expected,dict(expected=len(expected),observed=len(observed)))
    total_updates=sum(r['updates'] for r in runs)
    candidates=[json.loads(p.read_text()) for p in study.glob('worker-*/results/search-offset-*/selection.json')]
    require('Two equal-budget validation-only candidates',len(candidates)==2 and all(c['updates']==150 and not c['test_used'] for c in candidates),candidates)
    total_updates+=sum(c['updates'] for c in candidates)
    require('Scientific update budget used exactly once',total_updates==protocol['total_optimizer_updates'],total_updates)
    gpu=[]
    for worker in study.glob('worker-*'):
        status=json.loads((worker/'status.json').read_text())
        require(worker.name+' clean process exit',status['returncode']==0,status)
        identity=json.loads((worker/'results/identity.json').read_text())
        uuid=identity['gpu_uuid']
        rows=list(csv.DictReader((worker/'results/gpu-utilization.csv').open()))
        normalized=[{k.strip():v.strip() for k,v in row.items()} for row in rows]
        allocated=[row for row in normalized if uuid in row.get('uuid','')]
        utilization=[float(row['utilization.gpu [%]'].replace(' %','').replace('%','')) for row in allocated if '[Not Supported]' not in row['utilization.gpu [%]']]
        peaks=[r['peak_gpu_memory_bytes'] for r in runs if any((worker/'results').glob(f"{r['task']}-seed-{r['seed']}-{r['variant']}/summary.json"))]
        gpu.append(dict(worker=worker.name,job_id=identity['job_id'],uuid=uuid,name=identity['gpu_name'],
            samples=len(utilization),mean_utilization_percent=statistics.mean(utilization) if utilization else None,
            maximum_utilization_percent=max(utilization) if utilization else None,
            peak_cuda_memory_bytes=max(peaks) if peaks else None,
            allocated_tasks=[r['task']+'-'+str(r['seed'])+'-'+r['variant'] for r in runs if (worker/'results'/f"{r['task']}-seed-{r['seed']}-{r['variant']}/summary.json").exists()]))
        require(worker.name+' A100 CUDA telemetry',bool(utilization) and max(utilization)>0 and 'A100' in identity['gpu_name'],gpu[-1])
    require('Two distinct physical A100 GPUs used',len(gpu)==2 and len({g['uuid'] for g in gpu})==2,gpu)
    for p in study.glob('worker-*/results/*/summary.json'):
        run=json.loads(p.read_text())
        require(p.parent.name+' best/last/curve artifacts',all((p.parent/n).exists() for n in ['best.pt','last.pt','curve.csv']))
        if run['task']=='lm':
            require(p.parent.name+' full test targets',run['evaluation']['tokens']==55744)
            require(p.parent.name+' free generation',len((p.parent/'generation.txt').read_text())==415)
    require('Source bundle digest matches submission',all(hashlib.sha256((study/'source.tar.gz').read_bytes()).hexdigest()==json.loads(p.read_text())['source_bundle_sha256'] for p in study.glob('worker-*/manifest.json')))
    output=ROOT/'results/aggregate'
    (output/'gpu_evidence.json').write_text(json.dumps(gpu,indent=2))
    (output/'completion_audit.json').write_text(json.dumps(dict(all_passed=all(c['passed'] for c in checks),checks=checks),indent=2))
    print(json.dumps(dict(all_passed=all(c['passed'] for c in checks),failed=[c for c in checks if not c['passed']],gpu=gpu),indent=2))
    if not all(c['passed'] for c in checks): raise SystemExit(1)
