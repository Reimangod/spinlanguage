import hashlib,json,pathlib,subprocess
root=pathlib.Path(__file__).resolve().parent
cmd=json.loads((root/'attempt-2/manifest.json').read_text())['commands'][0][:-1]
remote="""hostname
sinfo -o '%P %a %l %D %G %t'
squeue -u reiwatanabe
sacct -j 5208,5209,5210,5211 -o JobID,JobName%36,State,ExitCode,Start,End,AllocTRES%80
scontrol show partition gpu-short
sacctmgr -P show qos gpu-short-qos format=Name,MaxWall,MaxJobsPU,MaxSubmitPU,MaxTRESPU
df -h /share/reiwatanabe /tmp
/share/reiwatanabe/ceo-bounded-source-20261007-v1/venv/bin/python -m pip freeze
"""
r=subprocess.run(cmd+[remote],capture_output=True,text=True)
(root/'final-evidence.txt').write_text(r.stdout+r.stderr)
print(r.stdout+r.stderr)
digests={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and p.name!='digests.json'}
(root/'digests.json').write_text(json.dumps(digests,indent=2))
