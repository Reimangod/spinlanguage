"""Resolve supplied bibliography against primary registries; preserve uncertainty."""
import concurrent.futures, hashlib, json, pathlib, re, urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1]
def fetch(url: str) -> bytes:
    req=urllib.request.Request(url,headers={'User-Agent':'magnonic-lm-research/0.1'})
    with urllib.request.urlopen(req,timeout=45) as response:
        return response.read()
def resolve(item: tuple[str,str,str]) -> dict:
    key,claimed,url=item
    record=dict(id=key,supplied_description=claimed,supplied_url=url,body_reviewed=False,
        implementation_basis=False)
    try:
        if 'doi.org/' in url:
            doi=url.split('doi.org/')[1]
            payload=json.loads(fetch('https://api.crossref.org/works/'+doi))['message']
            record.update(title=payload.get('title'),DOI=payload.get('DOI'),
                published=payload.get('published'),authors=payload.get('author'),
                license=payload.get('license'),registry='Crossref',status='metadata verified',
                article_type=payload.get('type'),source_links=payload.get('link'))
        else:
            payload=fetch(url).decode()
            title=re.search(r'<meta name="citation_title" content="([^"]+)"',payload)
            record.update(title=title.group(1) if title else None,status='arXiv abstract page retrieved',
                registry='arXiv',content_sha256=hashlib.sha256(payload.encode()).hexdigest())
    except Exception as exc:
        record.update(status='unverified',error=str(exc))
    return record
if __name__=='__main__':
    prompt=(ROOT/'codex_magnonic_lm_mvp_prompt.md').read_text()
    items=re.findall(r'^- \[([^]]+)\] (.+?), (https://(?:doi\.org|arxiv\.org)/[^\s]+)',prompt,re.M)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(resolve,items))
    (ROOT/'docs/reference_registry.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    print(json.dumps(dict(total=len(records),verified=sum(r['status']!='unverified' for r in records))))
    sources={
      'zenbaa-2403.17724v2.html':'https://arxiv.org/html/2403.17724v2',
      'koester-2507.15779v2.html':'https://arxiv.org/html/2507.15779v2',
      'spintorch-solver.py.txt':'https://raw.githubusercontent.com/a-papp/SpinTorch/main/spintorch/solver.py',
      'spintorch-focus.py.txt':'https://raw.githubusercontent.com/a-papp/SpinTorch/main/focus.py',
      'spintorch-readme.txt':'https://raw.githubusercontent.com/a-papp/SpinTorch/main/README.md',
      'tinyshakespeare-card.txt':'https://huggingface.co/datasets/karpathy/tiny_shakespeare/raw/main/README.md',
    }
    source_manifest=[]
    for name,url in sources.items():
        try:
            data=fetch(url); (ROOT/'third_party'/name).write_bytes(data)
            source_manifest.append(dict(file=name,url=url,sha256=hashlib.sha256(data).hexdigest()))
        except Exception as exc:
            source_manifest.append(dict(file=name,url=url,error=str(exc)))
    (ROOT/'third_party/manifest.json').write_text(json.dumps(source_manifest,indent=2))
    data=fetch('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
    (ROOT/'data/tinyshakespeare.txt').write_bytes(data)
    print('dataset',len(data),hashlib.sha256(data).hexdigest())
