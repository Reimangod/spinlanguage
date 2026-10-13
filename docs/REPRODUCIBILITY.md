# 実行と再現性の記録

## 固定された入力

要求仕様の原本はルートの `codex_magnonic_lm_mvp_prompt.md`。Tiny Shakespeareは1,115,394 bytes、SHA-256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`。train/validation/testは連続offset `[0,1003854,1059624,1115394]`。語彙65文字はtrainだけで作る。未知test文字があればエラーにする。

評価は各split内の重ならない32文字窓。test 55,770文字から55,744個の次文字targetを評価し、末尾25 targetを落とす。系列をまたぐ状態はゼロ。独立random training windowsなので位置/時刻をReadoutへ渡さない。合成はSeed `seed*100000+update`、validationは `5000000+seed*100+delay`、testは `9000000+seed*100+delay` と用途を分離する。

## 予算と比較

`configs/protocol_v1.json` は本学習より前に作成した。主LMは3 Seed×4モデル×600更新、合成は3 Seed×3モデル×300更新、外側2候補×150更新。合計10,200 optimizer updates。LM各runはbatch512×length32×600=9,830,400文字targetを訓練する。trainコーパスを全部含むsplitからsampleするので、epochを厳密な全量順次走査と呼ばない。

外側offset0/4は各150更新、Seed11、同じデータ/Readoutサイズ。full validation CEで選択する。testは構造やcheckpointの選択に使わない。比較の全best checkpointは固定予算内の最小validation CE。全test結果を残す。

## 環境

既存read-only環境はPython3.12.3、NumPy1.26.4、SciPy1.13.1、Qiskit1.0.2、CuPy13.6.0。`pip check` が成功した。今回専用 `/dev/shm/rei-magnonic-lm-20261014-v1/site` に公式CUDA12.4 wheelのPyTorch2.5.1をinstall。約5.1 GiB。A100 80GB、driver互換性と実CUDA reduction/FFT/BPTTを確認した。setup jobは `results/setup.log` に記録。

torch/NumPy seedを固定し、OMP/BLAS/MKL各1 thread、TF32を無効化した。CPU物理テストはMacの隔離 `.venv-test` でも実行した。CPU解析の `.venv-analysis` は図作成だけに使い、そのNumPy/Matplotlibの版は学習環境と区別する。libraryの完全lockは実環境freezeログが優先。

## GPU/保存証拠

本実験 `results/study-20261014-061533`、worker0 job5219、worker1 job5220。それぞれのmanifestに投入コマンド、source tar SHA-256、script digest、GPU割当、出力先、開始/終了時刻を保存。原実行ソースは同runの `source.tar.gz` に保存し、その後のdocs/CPU解析コードの追加と区別できる。

各workerは独立1-GPUジョブ。`nvidia-smi` は割当cgroupが見せるGPU群を列挙することがあるため、その一覧の全GPUを使用済みとは数えない。PyTorch `get_device_properties(0).uuid` と、同uuidのutilization時系列を照合する。ピークCUDA memoryはrunごとのtraining.jsonに記録。モデル学習のCUDA APIとtensor計算、実GPU uuid、利用率の3点から使用を確認する。

ジョブは最大2時間、学習の数値失敗/振幅0.20超過/予算終了で停止。ログインノードではSlurm情報取得、版の確認、軽いファイル管理だけを行う。2 workerに研究全体のrunを重複割当しない。

共有 `/share` は0空き、compute `/tmp` は100%だったので、大規模I/Oを避けた。依存+source+tiny corpus+小checkpointは専用RAM領域。出力見積100MBに対しcompute RAM filesystemは確認時約1008GB空き、ジョブmemory request16GB、専用依存5.1GBを用いた。Macへ完了run単位でtarをstreamし、最後に全resultsを保存。既存data/cacheを削除して空きを作っていない。

## 初版の失敗を残す

初期の汎用CuPy確認job5208/5209はcompute /tmp満杯で失敗、sacct FAILEDを確認してから修正版5210/5211を投入。六GPUの試験はQoSが同時2GPUへ制限しておりallocation拒否、計算未開始。

物理前処理の初版4substepは刻み誤差8.25%で採用せず、中点入力+台形観測へ修正。最終8substepは32tokenで64substepとの差0.936%。代表BPTTのCPU0.345s対GPU0.304s、約1.13倍。小バッチではGPUが遅かった。速度を物理device energy/Measurement Costと混同しない。

Whimsicalの構成図作成は無料枠上限で失敗。作成した空boardをtrashへ移し、ローカルSVG/PNGの概念図に切替。数値プロットは保存済みの実結果に基づく。
