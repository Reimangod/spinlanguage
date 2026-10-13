# 日本語結果報告 - YIG Magnonic LM MVP v1

## 実際に完成したものと結論

要求仕様を全文読み、空の作業ディレクトリから本チャット専用に物理モデル・コード・比較・自動逆設計・実データLMを作り、A100を2台使って実行した。設計だけで終了していない。主実験21 run、外側探索2 run、合計10,200 optimizer updatesを消化し、checkpoint・CSV/JSON・自由生成・図を保存した。

**今回の主結論は、内部Weight学習の言語予測上の利益を示せなかった、である。** 固定版と学習版の平均CEはほぼ同じで、平均では学習版がわずかに悪化した。合成記憶は改善と悪化が混在した。No-memoryは物理Reservoirよりよく、小GRUはさらによかった。これを成功の性能改善へ言い換えない。

一方で、局所電流→磁場→空間伝搬→帯域付き観測→CEの学習経路は実際に動き、電流も更新された。生成する文字モデルとしては動作しているが、文章品質は低く、現代LLMとの品質比較をする段階ではない。

## 自然言語の実測値

ここでの「実測値」はGPU/CPU上で実際に走らせたシミュレーションの数値であり、実機の測定ではない。Tiny Shakespeareのtrain由来65文字、90/5/5連続分割、32文字context、test 55,744 target、teacher forcing、各600更新・batch512。

| モデル | Test CE平均 [nats] | Seed標準偏差 | BPC平均 | Trainable parameter数 |
|---|---:|---:|---:|---:|
| 固定Reservoir | 3.201176 | 0.025409 | 4.618321 | 1105 |
| 内部Weight学習 | 3.204701 | 0.036075 | 4.623407 | 1113 |
| 記憶無効化 | 3.118932 | 0.017284 | 4.499667 | 1105 |
| 小GRU | 2.289583 | 0.005780 | 3.303171 | 2385 |

CE/BPCは小さい方がよい。固定/内部学習/No-memoryの線形Readoutは同じ1105パラメータ。追加内部Weightは8個。GRUは2385パラメータなのでparameter一致の勝負ではないが、この小さいデジタルモデルでも物理版を上回った。

| Seed | 学習版CE - 固定版CE | 解釈 |
|---|---:|---|
| 11 | +0.016002 | 悪化 |
| 22 | -0.002156 | 改善 |
| 33 | -0.003271 | 改善 |

平均paired差は +0.003525 nats。3 Seedしかなく、小効果の統計的確立はしていない。今回の固定予算の範囲では、学習版の優位性を採用しない。

診断的なdigital count baselineはunigram CE=3.361991、bigram CE=2.494808。主モデルはunigramより低いCEになったが、直前の文字IDだけを使うbigramにも及ばなかった。これらは外部digitalに文字IDを与える別モデルであり、物理Readoutにその情報を追加していない。

## 記憶タスクと選択性

8クラスDelayed Recall。最初のsymbol→blank delay→queryを与え、最後の観測だけからsymbolを当てる。validation/testは訓練と独立Seed。以下は各4096 test例、3 Seedの平均正解率。Selective CopyingやKey-Value Associative Recallの実験ではない。

| Blank遅延 | 固定 | 内部学習 | 記憶無効化 |
|---|---:|---:|---:|
| 1 | 34.01% | 46.09% | 11.87% |
| 4 | 33.16% | 37.61% | 12.33% |
| 8 | 33.94% | 37.57% | 12.56% |
| 16 | 25.50% | 29.54% | 12.66% |
| 24 | 20.59% | 13.39% | 12.10% |

chanceは12.5%。低減衰の波動状態が情報を残す場合はあるが、遅延別の応答・Seed依存が大きい。学習版Seed33・delay16は0%などの悪化条件も保存した。混合遅延CEを最小化する一つの制御パターンが、全遅延のargmax accuracyを同時に改善する保証はない。

## 自動逆設計の実行

入力ポートoffset0/4を各150更新・Seed11で学習し、validation CEで選択した。offset0=3.349845、offset4=3.324867。選ばれたoffsetは 4 セル。testを探索に用いず、選択後に主比較を最初から同予算で学習した。

これは製造前のポート配置探索である。製造後の内部Weightはコイル電流8個であり、交換定数やYIGのPMAを自由に書き換えたものではない。

## GPUと数値検証

| Worker / Slurm job | 物理GPU UUID | 利用率平均 / 最大 | runピークCUDA memory |
|---|---|---:|---:|
| worker-0 / 5219 | 27f75ca1-c8bb-be8c-3687-e60a5198bf05 | 31.6% / 39.0% | 114.0 MiB |
| worker-1 / 5220 | 766c2b71-228e-6498-9011-64c5408bc809 | 30.8% / 40.0% | 114.0 MiB |

両GPUはNVIDIA A100-SXM4-80GB。PyTorch CUDA tensor/FFT/BPTTを実行し、UUID別のnvidia-smi時系列と対応付けた。上記利用率は2秒間隔の観測で、すべてのkernelを測る精密profileではない。小モデルなのでA100の計算能力を使い切っていない。

Slurm最終照合: job5219は09分43秒、job5220は15分07秒、両方COMPLETED/ExitCode0。開始はともに2026-10-14 06:15:35 JSTで、同時利用を確認した。現在本実験の稼働/待機jobはない。

最終preflightではCPU/GPUのfeature最大差4.08e-6、制御gradient最大差1.28e-8、action norm相対差2.36e-5。電流autogradは有限差分テストを通した。駆動なし減衰、空間伝搬、因果性、状態リセット、分割、同一初期値、観測以外のReadout入力がないことを検査し、追加校正APIテストを含め13件がMacで成功した。

8 substeps/tokenは64 substepsとの差0.936% (32-token final state)。初版の粗い刻みは採用しなかった。学習済みcheckpointのtest subsetで刻みを半分にしたCE変化は固定Seed11で約-7.9e-5、学習版で約-1.7e-5 nats。この小さい差は結論が時間刻みだけで生じた説明を弱めるが、縮約物理の正しさの証明ではない。

代表BPTTはCPU0.345s/GPU0.304s、約1.13倍。小バッチではGPUが遅い。これはdigital simulationの速度であり、magnonic実機のJ/tokenやMeasurement Cost低減を示していない。

## 実験制約と追加感度

同じbest Weightにセンサ雑音、4/8-bit DAC、25 MHz帯域、300K thermal proxy、非線形無効、半分電流範囲を適用した。3 Seedの詳細は各runのablations.jsonと図。CPU再計算はSeed11だけでセンサ2本、±5% gain drift、10% extra coil crosstalk、半分刻みを追加した。test subsetの感度試験であり、条件の選択や再学習には使っていない。

センサ4→2本、Readout再学習なしでは固定版CEが+0.0654、内部学習版が+0.0627 nats悪化した。RF観測の資源削減には精度とのtradeoffがある。絶対電圧/実験SNR/RF・LNA・ADC/DC電源の電力は未校正なので数値的なエネルギー優位を主張しない。

## なぜ改善が出なかったか

観測された事実は、loss自体は下がる、8電流は動く、物理記憶は一部遅延で残るが、文字LMではNo-memoryがよい、である。16観測featuresに多数の文字と波動履歴を圧縮したため、現在文字の予測情報に過去の干渉が混ざる可能性がある。小さい駆動振幅ではKerr効果も弱く、任意の高次言語特徴を生成するほどの非線形性は期待しにくい。

以上は結果からの仮説であり、因果を確定した実験ではない。次に検証すべき順は、校正されたRF符号/観測ゲイン→linear/nonlinear/memory帯域のtrain時比較→短期語構造を保つ入力設計→少し長いcontext、である。同じtestで改善するまで条件を変更することは今回行っていない。巨大MLPを後付けして成功にしていない。

## コードと数式を理解する順序

1. config.py: T、m、A/m、J/m、s、Hzで全設定を読む。材料定数と制御上限を分ける。
2. physics.py: dispersion_rad_sがmode frequency、currents_Aが制限されたDC、local_field_Tが場map、wave_stepが空間時間発展。FFTは空間伝搬、Kerrは弱振幅の局所周波数補正。
3. step_token: 文字→固定RF符号→入力filter→波動→有限幅アンテナ→receiver filter→16 real観測。全セルの状態をreadoutへ渡さない。
4. models.py: 最後だけ線形分類。固定/内部学習の差はraw_currentへのgradientの有無。GRUは独立したdigital比較器。
5. data.py: 語彙はtrainから、次文字は右へ1だけshift、split境界を跨がない。合成の正解は先頭symbol。
6. training.py: CEをBPTTでReadoutと電流へ戻す。独立32文字窓の外でgradientは切れる。Adam/clip/有限差分照合の理由はTHEORY_AUDITにある。
7. study.py: 2候補を等予算で評価しvalidation最小を選ぶ。各workerは担当Seed/runだけを実行する。
8. aic_run.py: Slurm割当、source digest、利用率、scalar、checkpointのMac転送。SSH timeoutで同じjobを再投入しない。
9. diagnostics/postprocess/verify_study: checkpointのCPU再計算、実JSONからの図、全要求の証拠監査。calibration.pyは測定値が届いた時のAPIで、今回は使っていない。
10. RF符号表は65文字×4port×2成分=520 float32固定値（約2080 bytes）。デジタルのlookup/IQ変調は必要で、物理入力が無料とは扱わない。Readoutは1文字1040 multiply-addsと65-way softmaxで、そのほかRF/ADC/制御のhardware費用は未算定。

数式、近似、原著の確認箇所、API、テストを関数ごとに対応させた詳細は[THEORY_AUDIT.md](THEORY_AUDIT.md)。物性値の出典/仮定の相違、実験への最短経路は[EXPERIMENTAL_FEASIBILITY.md](EXPERIMENTAL_FEASIBILITY.md)。全67候補と17 repoの確認状態は[REFERENCES.md](REFERENCES.md)。

## 自由生成の実例（品質の低さも含む）

**fixed, Seed11, temperature0.8, 固定prompt以降は自己回帰:**

```text
First Citizen:
leeins lgm ilnr fh how ndh
rr
edria
eesndlmt  u  nsEool how: et o arl  eot  iyoea tnto Iurt tooraErt h. hio kasG
ra E diaeelaldyohs m,
hug tt toeRardrrfrea!t  ot;tc 
```

**trained, Seed11, temperature0.8, 固定prompt以降は自己回帰:**

```text
First Citizen:

eminl mem ilnr fn how nrndr weariactbent
ata m  usE ol hile et  saolh eot hi oha thyoourro tooraErhah. hio k
se
ra m diaeelaedyo top,
.rgstt
to!Ra
hNrfrealt  ot;tcf
```

**gru, Seed11, temperature0.8, 固定prompt以降は自己回帰:**

```text
First Citizen:
Herind me yiler falld; Kend,
If ricce sat
The math weill ill the sarl illd hind am tho nert thiran thang in kase
-a mid and ald she mach gret therard, freald bon; co
```

## 保存した成果物

- [比較図](../results/aggregate/language_comparison.png)、[記憶遅延](../results/aggregate/memory_delays.png)、[制約感度](../results/aggregate/experiment_ablations.png)。
- [Weight map](../results/aggregate/weight_distribution.png)、[mode waveform](../results/aggregate/mode_waveforms.png)、[概念図](../results/aggregate/concept.png)。概念図は実機layoutではない。
- [数値JSON](../results/aggregate/metrics.json)、[CSV](../results/aggregate/lm_metrics.csv)、[GPU証拠](../results/aggregate/gpu_evidence.json)、[実行監査](../results/aggregate/completion_audit.json)。
- 各runのbest.pt/last.pt/curve.csv/generation.txtは results/study-20261014-061533/worker-*/results/。元コードbundleとmanifestも保存。

独立LLG/実測校正、実ウェハの物性、量子輸送/イオン輸送、現代LLM級の品質、energy優位、Selective Copyingは未実行または未証明。それらをMVPの達成結果として混ぜない。今回の新規性候補は、実験制御に制約した内部磁場学習を文字LMへ接続する評価枠組みであり、先行研究をすべて排除した新規性主張はしていない。
