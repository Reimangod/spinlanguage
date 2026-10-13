# 理論・コード監査と読み解き方

この実装は**YIGを想定した、独自の空間的縮約モードモデル**である。PappらのSpinTorchの全LLG実装を複製したものではない。正しい単位と微分・因果性・刻み収束を検査しても、実機YIGのすべての振る舞いを再現した証明にはならない。未校正の係数は下記で区別する。

## 1. 出発点とモデルの階層

飽和磁化で規格化した磁化を `m=M/Ms` とすると、Gilbert減衰を含むLLGは

\[
\partial_t\mathbf m=-\frac{\gamma}{1+\alpha^2}
 [\mathbf m\times\mathbf B_{eff}
 +\alpha\mathbf m\times(\mathbf m\times\mathbf B_{eff})].
\]

`gamma` はrad/(s T)を使う。H[A/m]を使う定義とはmu0の位置が異なる。
交換磁場は `B_ex=(2A/Ms) ∇²m` [T]。反磁界はMaxwellの磁静的境界問題から決まる長距離項であり、交換と同じ項ではない。垂直飽和した平坦膜の一様内部磁場は `B_int=B0-mu0 Ms`。B0=350 mT、Ms=140 kA/mに対してB_int>0で、YIGに人工的なPMAを加えていない。

Pappらの原著プレプリント §5.1-5.3 と公式 `spintorch/solver.py` は、外場・交換・反磁界を含むLLG、FFTによる反磁界計算、RK4、autogradによる外場最適化を説明している。今回ここから採用したのは「磁場を物理Weightにし、微分を時間発展へ通す」という経路で、全LLGソルバーそのものではない。

Ms=140 kA/mは公式focus.pyと一致する。公式exch.pyはA=3.65 pJ/m、damping.pyはalpha=1e-4であり、今回のA=3.5 pJ/m、alpha=2e-4とは異なる。今回の値は近いオーダーのYIG代表値という仮定で、原著からそのままコピーした物性ではない。gamma=1.760859e11もSIの代表値を採った。実ウェハのVNA/BLS測定で置換する必要がある。

## 2. 線形分散を残す縮約

膜厚方向に一様な最低モードだけを残し、面内の1方向に周期Fourier基底を使う。空間点jは32個で、幅方向・厚み方向の高次モードは捨てる。`physics.dispersion_rad_s` は次の式を実装する。

\[
h_k=B_0-\mu_0M_s+\frac{2A}{M_s}k^2,\quad
P_k=1-\frac{1-e^{-|k|d}}{|k|d},\quad
\omega_k=\gamma\sqrt{h_k(h_k+\mu_0M_sP_k)}.
\]

P(0)=0は連続極限で与える。小さいkdでは `P≈kd/2-(kd)^2/6` を用いて差の桁落ちを避ける。最低厚みモードの横磁場を平均すると、波数方向の動的反磁界にはPが入り、それに直交する成分には入らない。二つの横成分の線形LLGを組み合わせると上記の平方根が出る。交換はhの中の一項、静的demagは `-mu0 Ms`、動的demagはPなので、同じ線形反磁界を二重に足していない。

これはuniform-mode diagonal近似の独自導出である。Kalinikos & Slavin (1986, DOI 10.1088/0022-3719/19/35/014) は分散理論の参考だが、取得を試みた原著本文はアクセス制限で未確認だった。したがって特定の式番号を推測して対応付けず、同論文の忠実な再現とは称さない。厚い膜や高次PSSWとのhybridizationが重要ならこの近似を使えない。

`omega_0` をRF搬送波に選び、その高速歳差運動を外した振幅psiを時間発展させる。実時間nsと包絡線を区別する。変化する搬送波を文字情報として使わない。

## 3. 学習可能な局所磁場と弱非線形

電流の無制限な最適化を防ぐため、各コイル電流は

\[
I_a=I_{max}\tanh\theta_a,\quad
\delta B_j=\sum_a C_{ja}I_a.
\]

thetaだけが物理内部の学習パラメータである。C[T/A]は固定のコイル応答で、今回のGaussianは要校正。Zenbaaらのプレプリント §II-A、§III-A、Fig.1 は、YIG・外部350 mT・独立DCコイル・±400 mAで約±2 mTを実験している。今回の8コイルの形状/寸法は同論文の49コイル実機と異なる。実験可能な制御機構を採り入れたのであって、同じ素子を再現したわけではない。

振幅方程式は概ね

\[
\dot\psi_j=-\mathcal F^{-1}[(\alpha\omega_k+i(\omega_k-\omega_0))\hat\psi_k]_j
 -i(\gamma\delta B_j+\nu|\psi_j|^2)\psi_j+F_j(u).
\]

`nu=gamma mu0 Ms/2` [rad/s]を初期仮定とした。円歳差の `mz≈1-|psi|²/2` を局所垂直demagへ入れるとこの符号・次数の周波数補正が得られる。しかしdipole-exchangeの真の非線形係数はモード・偏光に依存する。この係数は**独自の弱非線形提案・要校正**であり、Papp原著にある全反磁界非線形の置換精度は示せていない。非局所四次項、3/4-magnon scattering、Suhl不安定性、磁区、イオンゲートは実装していない。

十分小さい振幅を前提とし、学習中の最大振幅0.20超過で停止する。復元磁化の厳密な単位球を数値積分していないので、テストでいう保存量は弱振幅のmode action `sum |psi|²` であり、全LLGの磁化ノルム保存ではない。alpha>0なら駆動なしでactionは減る。alpha=0なら分割された位相回転とFourier伝搬でactionは保存される。

## 4. 積分・帯域・観測

`wave_step` は線形半ステップ→駆動半ステップ→局所位相の全ステップ→駆動半ステップ→線形半ステップを使う。FFT線形部とKerrだけの位相部は各々解析的で、大きい搬送波位相をEuler法で追わない。制御と非線形の非可換性は刻み誤差を残すので、時間刻み収束を別途検査する。

入力RFの一次フィルターは `du/dt=2pi f_in(u_target-u)`。入力値を半ステップ進めて中点のRFを駆動へ使い、残り半ステップを進める。初版ではステップ終端RFを使い、収束が悪かったので修正した。

アンテナはGaussian空間平均 `v_s=sum_j S_sj psi_j` とし、IQ復調後の一次フィルター `dy/dt=2pi f_sensor(v-y)` に入れる。wave_step前後の平均を駆動として近似積分する。磁化から誘導電圧への絶対ゲインは未測定なので、固定規格化ゲイン100を使う。入力場5 microtesla、帯域200/100 MHz、センサ寸法とこのゲインは今回の仮定である。

8 nsに2回だけサンプルし、4センサ×I/Q×2時点=16 features。Readoutが受け取るのはこの観測だけである。デバッグのmode waveform保存は学習・評価の入力ではない。固定Readoutは16→65のaffine、1105パラメータ、1040 multiply-adds/characterである。

## 5. 温度とノイズの範囲

基準のMs/alphaは室温YIGを想定し、T=300 Kは設定に保持する。基準比較では確率的熱励起を無効化する。熱アブレーションではcell volume=dx×width×thicknessと内部磁場からequipartitionを用いた局所OU振幅ノイズを加える。各実成分の定常分散を `kB T/(Ms V B_int)` とし、1刻みの係数を `sqrt((1-exp(-2 alpha omega0 dt))*variance)` とする。

これは完全な確率的LLGのFDT実装ではない。動的dipole couplingと楕円偏光を含む正確なmode covarianceを省略した熱雑音代理であり、実験SNRの予言には使えない。センサ加算ノイズは別の観測雑音として設定し、熱雑音と混同しない。

## 6. Loss、BPTT、比較

文字tを入力したあとの観測から文字t+1を予測する。`models.PhysicalLanguageModel` は `logits=W_readout feature+b` だけを計算する。softmax+CEは `F.cross_entropy` を使い数値的なlogsumexp処理をlibraryへ任せる。CEは自然対数、BPCは `CE/log(2)`。

固定版はraw_currentをfreezeしReadoutのみAdamで学習する。内部学習版は同じraw_current初期値・固定RFコード・Readout初期値から始め、raw_currentにも勾配を通す。`current_lr` と `readout_lr` を分ける。追加自由度は8個だけ。勾配clip=1、FP32/complex64、TF32を無効化する。複素autogradは実数LossのWirtinger勾配を扱うPyTorch機能に依存し、今回の電流微分は有限差分でも検査する。採用した2.5.1の実動作をテストしている。

訓練はランダムな独立32文字窓で、その窓内全時刻をBPTTする。窓をまたいで状態・勾配を渡さない。`detach_state` は将来の連続TBPTT用の明示APIで、主比較は毎窓リセット。長い履歴を学習した証拠ではない。生成時はprompt以降に実Tokenを見せず、サンプルした文字だけをfeedbackする。

No-memoryは各文字でmode/RF/receiver状態をゼロにする。このモデルも入力文字自体の情報は観測できるので、純粋なunigram baselineとは異なる。Delayed Recallではquery直前の記憶が消えているためクラス事前分布が上限になる。小GRUは同じ文字分割・更新回数を用いるが、物理版と完全なparameter budget一致はしていない。総パラメータを報告して物理優位の根拠にしない。

## 7. コード対応表

| ファイル/関数 | 採用理由と根拠 | 近似/制約 | 検証 |
|---|---|---|---|
| config.py / PhysicsConfig | SI値と実験制約を学習から分離 | 仮定値を含む | 周波数の単位・範囲 |
| physics.py / dispersion_rad_s | 空間波数・交換・薄膜磁静力学 | 最低厚みモード、周期1D | k=0周波数、伝搬 |
| gaussian_ports |有限幅励起/検出 | 要測定のGaussian | 規格化、局所磁場上限 |
| currents_A / local_field_T | Zenbaa §II-AのDCコイル機構 |8コイル応答は独自 |±400 mA/±2 mT |
| wave_step | 分散を保つ解析subflow | 局所Kerr・分割誤差 | 駆動なし減衰/action、刻み |
| step_token |帯域制限されたRF入力/観測 | 未校正IQ電圧 | 状態継続、No-memory |
| forward / detach_state |先読みなしの系列処理 |独立窓BPTT | 未来文字変更不変、切断 |
| models.py / PhysicalLanguageModel | Köster & Uchida §III.1/§Vの線形RC readoutを参考 |物理特徴は独自 | 同サイズ固定/学習 |
| TinyGRU |小デジタル比較器 |parameter不一致 |同分割/予算 |
| data.py / load_characters |学習由来語彙、連続分割 |作品間でなく文字offset分割 |境界/次文字シフト |
| recall_batch |独立Seedの8クラス記憶診断 |Selective Copyingではない |正解規則/Seed |
| training.py / optimize |有限差分検査済み経路のAdam/BPTT |モデル誤差を学習しうる |勾配clip、amplitude stop |
| evaluate_characters |teacher forcing CE/BPC |32文字窓の限定context |split別・重複しない窓 |
| evaluate_memory |遅延別test accuracy |簡単な1-symbol recall |別test Seed |
| generate |自己回帰の実行確認 |品質保証なし |実出力保存 |
| experimental_ablations |実験ノイズ/DAC/帯域の影響 |再学習せず設定変更 |固定と学習の両方 |
| study.py |等内側予算2候補、自動逆設計 |2候補のみ、1探索Seed |validationのみ選択 |
| aic_setup.py / aic_run.py |隔離依存/Slurm/2worker |RAM領域は永続しない |job ID、digest、CUDA利用率 |
| audit_sources.py / software_audit.py |原典と公開状態の監査 |本文未確認を残す |URL/digest/失敗保存 |
| calibration.py / apply_calibration |実測/field-solverのC[T/A]読込 |v1実験では不使用、校正データなし |単位/由来/field経路 |
| diagnostics.py / replay / ce_cpu |checkpointの独立CPU再計算 |少数固定test窓、選択しない |半分刻み/センサ削減/ドリフト |
| generate_checkpoint.py |保存Weightから実行するCLI |信頼したlocal checkpoint限定 |CPUで自由生成再実行 |
| count_baselines.py |損失低下が文字頻度/直前文字で説明されるか |digitalに直接文字ID、物理との同サイズ比較ではない |同じ55,744 test target |
| postprocess.py / plot_waveforms.py |実結果を図へ変換 |平均/Seed SD、小さいn |元JSON/CSV/NPZとの対応 |
| plot_concept.py |構成の説明 |概念図、実測layoutではない |数値図と別表示 |
| verify_study.py |全run/予算/GPU使用の完了監査 |物理真実性を証明しない |21run、2候補、10,200更新 |
| snapshot_versions.py / build_reference_notes.py |版・digest・人が読める監査 |測定versionと未確認を区別 |freeze/registry |
| write_results_report.py |実データから日本語報告を生成 |未証明を結論へ混ぜない |completion auditが通らなければ停止 |
| package_results.py |私用のcode/checkpoint/report一式 |installed環境・原著PDFを除外、source bundle内に当時の入力あり |ZIP CRC・SHA-256 |

## 8. 原典との比較で残る限界

Pappは全LLG、吸収境界、非線形反磁界、微細磁石の学習を扱った。今回は周期境界、1Dのuniform厚み包絡線、DCコイルを使う別モデルである。Zenbaaは49コイルを実測feedbackで最適化したが、今回は8制御のモデル上BPTTである。Köster & UchidaはデジタルRC/Attention/Transformerであり、今回のYIG内部Weightの学習を実証していない。これらの差を論文再現と呼ばない。

採用していない総説やMTJ論文は文献候補として残し、未知の機構を加える根拠にはしない。改善した場合も「小さいシミュレーション内で、追加8自由度を学習した効果」に限定する。実機優位や新規性の確定には校正・独立再現・先行研究精査が必要である。
