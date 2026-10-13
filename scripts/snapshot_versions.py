"""Save versions and digests without reading SSH keys or altering environments."""
import hashlib,json,pathlib,re,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]
lock=[]
for line in (ROOT/'results/setup.log').read_text().splitlines():
    if line.startswith('Successfully installed '):
        for entry in line.split('Successfully installed ',1)[1].split():
            match=re.match(r'(.+)-(\d[^ ]*)$',entry)
            if match:lock.append(match[1]+'=='+match[2])
lock.extend(['numpy==1.26.4','scipy==1.13.1','pytest==9.1.1','python==3.12.3'])
(ROOT/'results/cuda-environment-lock.txt').write_text('\n'.join(sorted(set(lock)))+'\n')
for env in ('.venv-test','.venv-analysis'):
    result=subprocess.run([str(ROOT/env/'bin/python'),'-m','pip','freeze'],capture_output=True,text=True,check=True)
    (ROOT/'results'/f'{env[1:]}-lock.txt').write_text(result.stdout)
digests={}
for directory in ('src','scripts','tests','configs','docs'):
    for path in (ROOT/directory).rglob('*'):
        if path.is_file() and '__pycache__' not in str(path):
            digests[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
for name in ('README.md','requirements.txt','pyproject.toml','codex_magnonic_lm_mvp_prompt.md'):
    digests[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
(ROOT/'results/final-source-digests.json').write_text(json.dumps(digests,indent=2))
print('version locks and digests',len(digests))
