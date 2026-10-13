"""Private deliverable: original code, reports, numeric results and checkpoints.

Excludes installed environments and standalone third-party article/data files.
Original study source bundle retains its input corpus for this private delivery.
"""
import hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
target=ROOT/'magnonic-lm-mvp-v1.zip'
paths=[]
for directory in ('src','tests','scripts','configs','docs','results','aic-preflight-20261014'):
    for p in (ROOT/directory).rglob('*'):
        if p.is_file() and '__pycache__' not in str(p):paths.append(p)
paths.extend(ROOT/name for name in ('README.md','requirements.txt','pyproject.toml','codex_magnonic_lm_mvp_prompt.md'))
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(set(paths)):archive.write(path,'magnonic-lm-mvp-v1/'+str(path.relative_to(ROOT)))
with zipfile.ZipFile(target) as archive:
    if archive.testzip() is not None:raise RuntimeError('Corrupt zip')
record=dict(file=str(target),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    files=len(set(paths)),checkpoint_files=sum(p.suffix=='.pt' for p in paths))
(ROOT/'results/delivery.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
