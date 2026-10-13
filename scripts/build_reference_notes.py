"""Human-readable bibliography distinguishes metadata from implementation evidence."""
import json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
refs=json.loads((ROOT/'docs/reference_registry.json').read_text())
selected={'P02':'原著preprint全文・Methods §5.1-5.3・公式solver確認。LLG/場学習の考え方の根拠。独自縮約を再現と称さない。',
          'A16':'原著preprint本文・Methods §III-A・Fig.1確認。YIGの局所DCコイル更新の根拠。掲載版の補足PDF取得はDNS失敗。',
          'P01':'原著preprint本文 §III.1/§V/§VI、掲載書誌・公式repo確認。線形RC readout/文字LM評価を参考。'}
lines=['# REFERENCES - 確認範囲を区別した文献一覧','',
       '2026-10-14に原典URLとCrossref/arXivへ照合。書誌情報の一致は本文の全検証を意味しない。全67件の完全再現は行っていない。',
       '','## 主に使った根拠','']
for r in refs:
    if r['id'] in selected:
        title=r.get('title');title='; '.join(title) if isinstance(title,list) else title
        lines.extend([f"- **{r['id']} {title}**: [{r['supplied_url']}]({r['supplied_url']})。{selected[r['id']]}"])
lines.extend(['','追加の分散式はLLG+uniform厚みの磁静力学から独自導出した。Kalinikos & Slavin (1986), DOI [10.1088/0022-3719/19/35/014](https://doi.org/10.1088/0022-3719/19/35/014) は参考文献だが本文アクセス制限のため式番号まで未確認。',
    '', 'PyTorch 2.5.1の公式 [complex autograd文書](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/complex_numbers.rst) と実動作でFFT/複素勾配を確認。',
    '', '## 全候補の書誌確認','', '| ID | 確認した正式題名 | URL | 確認範囲 |','|---|---|---|---|'])
for r in refs:
    title=r.get('title');title='; '.join(title) if isinstance(title,list) else title or '未取得'
    status='選定原著本文確認（版は上記）' if r['id'] in selected else r['status']+' / 本文・実験種別・適用項は未精査、実装根拠にしない'
    lines.append(f"| {r['id']} | {title.replace('|','/')} | [原典]({r['supplied_url']}) | {status} |")
lines.extend(['','正確な著者・刊行日・DOI・登録licenseは `reference_registry.json` に保存。Crossref licenseは記事の権利でありコードlicenseとは別。',
    '', '題名が曖昧だったA07/A08/A14/A19/A23/A24はregistryの正式題名に修正。A11は取得したarXivでは `An all-magnonic neuron with tunable fading memory`。BoTorchの公式repoはmeta-pytorchへ移転している。',
    '', '## ソフトウェア確認','', '| ソフト | 正式repo | 公開license情報 | 利用 |','|---|---|---|---|'])
for r in json.loads((ROOT/'docs/software_registry.json').read_text()):
    name=r['supplied_url'].split('/')[-1]
    license=(r.get('license') or {}).get('spdx_id','未判定')
    lines.append(f"| {name} | [{r.get('canonical_url',r['supplied_url'])}]({r.get('canonical_url',r['supplied_url'])}) | {license} | 調査のみ、導入しない |")
lines.extend(['','SpinTorchには `licenses/LICENSE_Wavetorch.md` があるが、APIが全repoのlicenseを判定していない。今回solverのソースをコピーして利用せず独自実装した。LAERCのlicenseも自動判定不能で、導入していない。',
    '', '[NeuralMag公式docs](https://docs.neuralmag.org/) はMIT、JAX/PyTorch、LLG/交換/demag等を確認。magnum.np公式docsは取得時502で、互換性試験は行っていない。MuMax3などの候補も実行互換性を実証していない。',
    '', '## データと取得失敗','',
    'Tiny ShakespeareはKarpathyの[原配布](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare)を取得しSHA-256を記録した。HF cardのlicense欄は `More Information Needed` で、明示licenseを確認できない。原文はShakespeareのpublic-domain作品だが、配布ファイルの再配布権を勝手に断定しない。',
    '', '掲載Natureページはブラウザ取得で認証redirect、Papp本文はarXiv PDFを取得して全文抽出、ZenbaaはarXiv HTMLを確認した。Zenbaa掲載補足PDFはDNS失敗。過去の試作bundle/pdf/xlsxはこの作業ディレクトリに存在せず、採用していない。'])
(ROOT/'docs/REFERENCES.md').write_text('\n'.join(lines)+'\n')
print('wrote references',len(refs))
