"""Inspect public official repositories without installing unrelated solvers."""
import concurrent.futures,hashlib,json,pathlib,re,urllib.request
from audit_sources import ROOT,fetch

def inspect_repo(url: str) -> dict:
    slug=url.split('github.com/')[1].rstrip('.,/')
    try:
        metadata=json.loads(fetch('https://api.github.com/repos/'+slug))
        record=dict(supplied_url=url,canonical_url=metadata['html_url'],public=not metadata['private'],
            license=metadata.get('license'),default_branch=metadata['default_branch'],
            description=metadata.get('description'),status='official repository metadata retrieved',
            compatibility='not installed or tested; only selected PyTorch runtime used')
        commits=json.loads(fetch('https://api.github.com/repos/'+slug+'/commits?per_page=1'))
        record['head_sha']=commits[0]['sha']
        return record
    except Exception as exc:
        return dict(supplied_url=url,status='unverified',error=str(exc))

if __name__=='__main__':
    prompt=(ROOT/'codex_magnonic_lm_mvp_prompt.md').read_text()
    urls=sorted(set(re.findall(r'https://github.com/[\w.-]+/[\w.-]+',prompt)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        records=list(pool.map(inspect_repo,urls))
    (ROOT/'docs/software_registry.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    for name,url in [('papp-2012.04594.pdf','https://arxiv.org/pdf/2012.04594'),
                     ('zenbaa-supplement.pdf','https://static-content.springer-cdn.com/esm/art%3A10.1038%2Fs41928-024-01333-7/MediaObjects/41928_2024_1333_MOESM1_ESM.pdf')]:
        try:
            data=fetch(url)
            if not data.startswith(b'%PDF'): raise ValueError('not PDF')
            (ROOT/'third_party'/name).write_bytes(data)
            print(name,len(data),hashlib.sha256(data).hexdigest())
        except Exception as exc: print(name,str(exc))
    print('repos',len(records),'verified',sum(r['status']!='unverified' for r in records))
