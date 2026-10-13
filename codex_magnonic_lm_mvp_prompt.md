# Codex実装依頼：Experiment-Aware Inverse Design of a Trainable YIG/Magnonic Language Model — MVP

あなたは、スピントロニクス・マグノニクス、微分可能物理シミュレーション、逆設計、時系列機械学習に精通した研究エンジニアです。**調査・設計で終わらず、コードを実装し、実際に学習・逆設計・比較評価を実行して結果を残してください。** 私への逐次的な承認要求は避け、合理的な選択を自律的に進めてください。物理モデルの最終選択と細かい理論・アルゴリズム設計はあなたの責任で行ってください。

## 0. 研究目的と境界

**最終目標：** 実験で制御・観測できることを意識したYIG優先のSpintronic/Magnonic Reservoirをシミュレーションし、**物理内部のWeight／制御パラメータを学習または逆設計**することで、小さな生成型Language Modelを成立させるMVPを作る。重要なのは、物理内部の学習が固定Reservoirに対して有益で、外部MLPやデジタル処理を大きくせずに言語予測性能・記憶能力が改善するかを科学的に検証すること。

**必須要件：**
- 材料は可能な限り**YIG（Yttrium Iron Garnet）**を優先。YIGが選択した機構と整合しない場合は理由を示し、適切な代替案と比較する。**YIGにPMAやVCMAが当然備わると仮定しない。** 実験的に実現可能な入力・制御を優先する。
- 現時点では**シミュレーションのみ**でよい。**MVPなので物理ソルバー／シミュレーターは1種類を主軸にしてよく、支配方程式も必要最低限にとどめる。** 本格的なMTJ量子輸送・イオン輸送・3D多物理連成などは、必要性を証明できない限り実装しない。
- 「スピン波・マグノン」を名乗るなら、伝搬・モードなどを表現できる**空間自由度を持つLLGまたは正当化されたマグノン結合モード縮約**を使用する。単一macrospinだけなら「磁化ダイナミクスMVP」と正直に明示し、伝搬マグノンを再現したと主張しない。
- **内部Weightの実験的な更新可能性**を区別する。磁性体の形状や製造後に固定される材料定数は「製造前の構造探索」、実際に可変な局所磁場・電流・プログラム可能な磁化状態・根拠ある電圧制御等は「製造後に学習可能なWeight」として扱う。直接任意のPMAや交換定数を書き換えられると仮定しない。
- 物理Reservoirへの入力、状態履歴、測定可能な出力、外部のReadout、学習対象をはっきり分ける。**物理内部にない情報をReadoutへ渡さない／未来のTokenを使わない。**
- 外部Readoutは**線形＋Softmax**を最初の本命とし、MLPの強化で物理内部の不備を隠さない。より複雑な外部モデルは比較対象・アブレーションでのみ必要に応じて使う。
- 最初の目標は現代LLM並みの品質ではない。**小語彙の自己回帰的な次Token予測を実際に学習し、自由生成を実行すること**、内部Weight学習の効果を公平に評価すること。

## 1. 自律的に選定してほしい技術

以下は選択肢であり、採用を強制しない。**Codex自身が原著・公開コード・公式ドキュメントを調べ、物理的整合性、計算負荷、実験的な制御可能性を根拠に最小構成を選択すること。**

- 物理：LLG／確率的LLG、またはLLGから導ける縮約モードモデル。必要なら交換・反磁界・異方性・減衰・局所入力、熱ノイズなど。実装する全項の根拠を明示。
- 材料：YIGを第一候補。低減衰の伝搬波、RF励起・検出、局所磁場／構造パターン等が候補。YIGで実現根拠の薄い局所VCMA・PMAの自由更新は採用しない。代替候補としてGa:YIG、CoFeB/MgO、SAF等を検討してもよい。
- 物理Weight／探索変数の**例**：局所コイル電流または磁場、静的磁場パターン、デバイス形状・ポート配置、散乱体の配置、異方性、膜厚、励起周波数・振幅・位相、局所制御電圧、測定位置、結合強度、減衰、温度など。材料定数と実機で可変な制御量を区別する。
- 最適化：微分可能物理系のBPTT／Adjoint／checkpointing、勾配不要のBayesian Optimization・進化探索・SPSAなどから、**内側のWeight学習と外側の構造探索に最適なものを選ぶ**。最初から大規模な探索をしない。
- 目的関数の**例**：次文字／次Tokenのvalidation cross-entropy（主目的）、Bits per Character、Selective Copying、Delayed Recall、Associative Recall、非線形記憶、内部Weight学習による改善、固定した予測損失に必要な外部Readout計算量、センサ数、ノイズ耐性、制御コスト／推定エネルギー。代理指標と最終目的を混同しない。
- データセットの**例**：①Seedを分けて生成するSelective Copying / Key–Value Associative Recall、②文字単位Tiny Shakespeare（MVPの生成型LM）、③余裕があればTinyStoriesの小規模subset。最初からOpenWebText全量や5万語彙は不要。**合成タスクだけで「Language Modelを完成」としない。**
- 比較対象の**例**：固定ランダムReservoir＋同一線形Readout、内部Weight学習版＋同一線形Readout、No-memory／memory-disabled、外部Readoutだけ強化した版、小さなGRU・ESN・Transformer等。公平なトークン化、学習データ量、評価条件を守る。

## 2. 実験Awareの設計原則

実機を構築しなくても、以下を可能な範囲で最初から含めるか、少なくとも設定可能な実験制約として分離する：
- SI単位・物性値の出典、温度、熱揺らぎ／減衰、時間刻み収束、入力電極・コイルの帯域と振幅上限。
- 実際に設定できる制御信号にのみ学習自由度を与える。電圧／電流上限、DAC分解能、クロストーク、制御更新速度、素子ばらつき・ドリフト、書き換えコスト。
- 実験で測れる観測量のみからReadoutを作る（RFアンテナの帯域付き検出電圧、平均磁化やTMRなど、選定した素子に即した測定モデル）。全セルの完全な磁化やノイズのない全モードを特権的に出力しない。
- 実測値がないパラメータは**仮定／要校正**と明記し、根拠のない実験精度やエネルギー優位を主張しない。
- 必要なら校正ファイル／装置データの読込APIだけ実装し、実測データがないのに擬似データを「実測」と称さない。
- ノイズあり／なし、制御量子化あり／なし、センサ数、物理Weight固定／学習を分けて比較する。

## 3. A100 GPUを必ず2台利用

計算環境には**NVIDIA A100 GPUを2台**使用したい。実行開始時に `nvidia-smi`・PyTorchのCUDA環境・各GPU名／VRAMを検査し、結果をログに残す。

- **両GPUを実際に使うこと。** 外側の候補評価をGPU 0とGPU 1に並列配置するか、必要に応じてPyTorch DistributedDataParallel等を利用。単に2台見えているだけでは利用したと見なさない。
- 物理シミュレータが単一GPUしか扱えない場合も、**独立した複数候補・Seed・物理計算を2プロセスで分配**する構成を優先する。
- 各runにGPU ID、割当タスク、ピークメモリ、実行時間、実際の利用率を記録し、最終報告で両GPUの稼働を示す。
- A100を利用できない場合は、**その事実を明記してCPU／単GPUでSmoke Testまで実施し、本番2GPU実験を実施済みと偽らない。** 必要な環境と本番実行コマンドを残す。
- 数値安定性を優先。低精度が物理積分や勾配を破壊する場合はFP32/FP64を使い、混合精度は根拠ある箇所のみ。OOMや異常勾配を防ぐ。

## 4. 実装と実行の順序

1. 既存のリポジトリ／添付資料があれば調査。下記の論文とソフトウェアについて**題名・DOI・公開状態・実験かシミュレーションか・適用できる物理項・ライセンス・互換性**を確認する。誤引用・利用不能なコードは明示して代替する。全論文の完全再現をMVPの条件にはしない。
2. 1ページ程度の**Architecture Decision Record（ADR）**を書く。どの材料・状態・入力・制御Weight・観測量・主シミュレータ・最適化・データセットを採用したか、採用しなかった選択肢と理由を書く。過剰な物理・ツール統合は避ける。
3. **まず最小のForwardモデルを作り、物理テストを通す。** 時間刻み・単位・磁化ノルム・駆動なし減衰・境界・初期状態・Seed・熱・勾配の有限差分照合などを選定モデルに応じて検証する。
4. 内部Weightについて**実際の制御信号 → 物理効果 → 時間発展 → 観測 → Loss**の計算経路を作り、勾配法ならautograd/adjointと有限差分を照合する。勾配法が妥当でないなら、その根拠と代替法を示す。
5. **固定Reservoir vs 学習可能Reservoir**の比較を最優先して、Selective Copying／Associative Recallなどで記憶・選択性を評価。共通のReadoutとデータ分割を使い、複数Seedで比較する。
6. 次に**Tiny Shakespeare等の実データで文字単位のNext-token LMを学習し、Teacher Forcingのtest loss/BPCと自由生成の両方を実行する。** 小さなサブセットで動作させてから全量へ進む。
7. 余力があれば、**形状・ポート・制御位置等の少数候補の外側自動探索**を実行。各候補を同じ内側学習予算で評価し、validationで選ぶ。性能のよい候補をtestで最後に評価する。探索でtestを使わない。
8. 実験Awareアブレーション（ノイズ、帯域、DAC、ばらつき、制御域）と、外部Readout計算量を評価。
9. テスト、学習、評価、サンプル生成、図表出力を**実際に実行**し、checkpoint・CSV/JSON・プロット・ログ・失敗結果を保存する。研究結果が弱ければ正直に記録し、架空の精度やグラフを作らない。

## 5. コード品質・理解・原著整合性（非常に重要）

- **可読性を最優先**。Python中心、意味のあるファイル・関数名、型ヒント、dataclassによる明示的Config、小さいモジュール、docstring、再現可能なCLI、依存関係の固定、単位を名前や説明に記載。物理とML、データ、探索、評価、可視化、回路モデルを分離。不要な抽象化や巨大な1ファイルは避ける。
- **作成する全ての主要コード・関数・物理項・学習ロジックに「なぜ採用したか」を理解して説明する。** 単にLibraryのAPIが動いたから採用することを禁じる。論文由来の実装、公式ドキュメント由来、MVPの独自仮定を区別する。
- `THEORY_AUDIT.md` 等に**コードモジュール／物理機構／支配方程式／論文の該当Section・式・図／公式API／近似／単位／制約／テスト**を対応付ける。原著が支持しない実装なら「提案・仮定」と明記し、決して論文の再現と称さない。
- 勾配の有無にかかわらず、**なぜその最適化が正しいか、別手段の方がよくないか、コストと安定性**を自ら批判的に検討する。特にYIGの異方性、PMAの実装可能性、交換と反磁界の二重計上、マグノンとmacrospinの混同、減衰と長文記憶の混同をチェック。
- `EXPERIMENTAL_FEASIBILITY.md` に実験で書き込めるWeight、必要装置、測定可能な信号、未知の校正パラメータ、実装難易度・主要な障害を記載。
- 「損失が下がった」の原因を解明する：記憶なしベースライン、内部Weight固定、Readoutのみ学習、入力符号化の情報漏洩、データのリーク、ランダムSeed、parameter budgetなどの切り分け。
- 言語予測のデータ分割、系列境界の状態リセット、先読みなしのcausal forward、BPTT勾配の切断点を単体テストする。単なる合成タスク正解率を自然言語LMの成功と混同しない。
- 外部の実測データ・API・ダウンロードが使えなければ、どこがブロックされたか報告し、**独立に完了できる最小実験を最後まで進める。**

## 6. 期待する成果物

- **動くRepository**：`README.md`、環境構築、起動コマンド、設定ファイル、`src/`、`tests/`、`scripts/`、`configs/`、`results/`、`docs/` を整理。必要なら構成を簡略化。
- **データセット生成・取得**：Synthetic memory、文字LMデータの準備。データ使用ライセンスを確認。
- **MVP物理モデル**：実験Awareな入力・制御・観測、内部Weightを持つ時間発展、トレーニング可能なモデル、チェックポイント。
- **逆設計実行結果**：固定vs学習、複数Seed、各学習曲線、記憶遅延別精度、言語test CE/BPC、サンプル文章、探索結果、失敗・制約。
- **可視化**：モデル全体と状態の流れ、形状・Weight分布、磁化／モード波形、記憶遅延・予測損失・物理ノイズと性能の関係。図は計算結果と概念図を明確に区別。
- **監査資料**：`REFERENCES.md`、`THEORY_AUDIT.md`、`EXPERIMENTAL_FEASIBILITY.md`、`REPRODUCIBILITY.md`、`LIMITATIONS.md`、`DECISIONS.md`。文献の確認範囲を示す。
- **GPU証拠**：A100×2の検出ログ、各GPUへ割り当てた計算・実行時間と利用状況。
- **最後の日本語報告**：実際に動いたもの、測定した値、予測との差、採用理論の根拠、新規性になり得る点、まだ示せない点、実機へ進む最短ルートを明快に説明。

**完了条件：文献リストや設計書だけで終わらず、最小の物理時間発展＋学習・逆設計＋固定Reservoir比較＋実データの次Token予測を実行し、数値・図・checkpointを提出すること。** 到達できない項目は「未実行」「失敗」「環境不足」と明示する。

---

# 参考文献（このチャットで登場したもの）

以下は**関連研究の調査リスト**であり、すべてを実装するという意味ではない。DOI/URLが不正確、題名変更、プレプリントと掲載版が異なるなどの可能性があるため、**Codexが原著・公式Repositoryに照合して正しい書誌情報に修正すること**。調査済・未確認・コード公開有無を区別し、実装根拠にするのは内容を確認した文献だけにする。簡潔に分類し、最終採用する少数の文献は本文／Methods／Supplementary／公式コードまで確認すること。

## A. Reservoir Computing・Language Model・評価理論

- [P01] Köster & Uchida, **Reservoir Computing as a Language Model** (2026), https://doi.org/10.1103/sd11-x3ny — 固定RC・AERC・LAERC、次文字／次Token生成。
- [P33] Jaeger & Haas, **Harnessing Nonlinearity: Predicting Chaotic Systems and Saving Energy in Wireless Communication** (2004), https://doi.org/10.1126/science.1091277 — ESN/RC基礎。
- [P34] Tanaka et al., **Recent Advances in Physical Reservoir Computing: A Review** (2019), https://doi.org/10.1016/j.neunet.2019.03.005 — Physical RCのレビュー。
- [P35] Sun et al., **In-sensor Reservoir Computing for Language Learning via Two-dimensional Memristors** (2021), https://doi.org/10.1126/sciadv.abg1455 — 物理RC＋言語関連処理。
- [P36] Gu & Dao, **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** (2023/24), https://arxiv.org/abs/2312.00752 — 入力選択的状態更新。
- [P37] Peng et al., **RWKV: Reinventing RNNs for the Transformer Era** (2023), https://doi.org/10.18653/v1/2023.findings-emnlp.936 — Recurrent LM。
- [P38] Köster et al., **Attention-enhanced Reservoir Computing** (2024), https://doi.org/10.1103/PhysRevApplied.22.014039 — 可変Readout。
- [P41] Liang et al., **Physical Reservoir Computing with Emerging Electronics** (2024), https://doi.org/10.1038/s41928-024-01133-z — 電子Physical RCレビュー。
- [P42] **Topological Magnetic and Ferroelectric Systems for Reservoir Computing**（書誌確認）, https://doi.org/10.1038/s42254-024-00729-w — 関連レビュー候補。
- [P43] **Metrics for Spin-based Computing**（書誌確認）, https://doi.org/10.1038/s42254-025-00918-1 — ハードウェア指標。
- [A01] Dambre et al., **Information Processing Capacity of Dynamical Systems** (2012), https://doi.org/10.1038/srep00514 — RC容量の理論。
- [A02] Vaswani et al., **Attention Is All You Need** (2017), https://arxiv.org/abs/1706.03762 — 生成型LMの比較対象。
- [A03] Hatamizadeh et al., **Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention** (2026), https://arxiv.org/abs/2605.22791 — Write/Erase/Read操作の参考。
- [A04] Eldan & Li, **TinyStories: How Small Can Language Models Be and Still Speak Coherent English?** (2023), https://arxiv.org/abs/2305.07759 — 小規模LM用データ。

## B. 多層Reservoir・Physical Neural Network・学習

- [P04] Gallicchio et al., **Deep Reservoir Computing: A Critical Experimental Analysis** (2017), https://doi.org/10.1016/j.neucom.2016.12.089 — DeepESN。
- [P05] Liu et al., **Multilayer Reservoir Computing Based on Ferroelectric α-In₂Se₃ for Hierarchical Information Processing** (2022), https://doi.org/10.1002/adma.202108826 — 強誘電体多層RC。
- [P06] Lupo et al., **Deep Photonic Reservoir Computer Based on Frequency Multiplexing with Fully Analog Connection Between Layers** (2023), https://doi.org/10.1364/OPTICA.489501 — アナログ層間接続。
- [P07] Shen et al., **Deep Photonic Reservoir Computing Recurrent Network** (2023), https://doi.org/10.1364/OPTICA.506635 — 多層光RC。
- [P08] Nishioka et al., **A High-performance Deep Reservoir Computer Experimentally Demonstrated with Ion-gating Reservoirs** (2024), https://doi.org/10.1038/s44172-024-00227-y — 実機多層RC。
- [P09] Stenning et al., **Neuromorphic Overparameterisation and Few-shot Learning in Multilayer Physical Neural Networks** (2024), https://doi.org/10.1038/s41467-024-50633-1 — 多層物理応答。
- [P20] Wright et al., **Deep Physical Neural Networks Trained with Backpropagation** (2022), https://doi.org/10.1038/s41586-021-04223-6 — Physics-aware Training。
- [P21] Nakajima et al., **Physical Deep Learning with Biologically Inspired Training Method** (2022), https://doi.org/10.1038/s41467-022-35216-2 — gradient-free/DFA系学習。

## C. PMA・MTJ・磁気状態・学習可能な物理Weight

- [P03] Pereira et al., **Trainable Neuromorphic Spintronic Hardware via Analog Finite-difference Gradient Methods** (2026), https://doi.org/10.1038/s41467-026-73624-w — 実MTJの活性化・有限差分勾配。
- [P10] Nomura et al., **Reservoir Computing with Dipole-coupled Nanomagnets** (2019), https://doi.org/10.7567/1347-4065/ab2406 — VCMA／結合磁石RC。
- [P11] Taniguchi et al., **Spintronic Reservoir Computing without Driving Current or Magnetic Field** (2022), https://doi.org/10.1038/s41598-022-14738-1 — VCMA・磁化緩和。
- [P12] Edwards et al., **Passive Frustrated Nanomagnet Reservoir Computing** (2023), https://doi.org/10.1038/s42005-023-01324-8 — PMA磁石とフラストレーション。
- [P13] Zeng et al., **Radio-frequency-modulated Artificial Synapses Based on MTJs with PMA** (2024), https://doi.org/10.1103/PhysRevApplied.21.014020 — RF Weight。
- [P14] Xiang et al., **An Antiferromagnetic Neuromorphic Memory Based on Perpendicularly Magnetized CoO** (2024), https://doi.org/10.1021/acs.nanolett.4c02340 — 反強磁性Fading Memory。
- [P15] Das et al., **A Magneto-ionic Synapse for Reservoir Computing** (2025), https://doi.org/10.1103/PhysRevApplied.23.054043 — イオン・磁気状態。
- [P16] Rajib et al., **Magneto-Ionic Physical Reservoir Computing in Perpendicularly Magnetized Heterostructures** (2025), https://doi.org/10.1021/acs.nanolett.5c03889 — PMA/イオン履歴。
- [P17] Kurenkov et al., **Perpendicular-anisotropy Artificial Spin Ice with Spontaneous Ordering** (2025), https://doi.org/10.1038/s44172-025-00499-y — PMA磁石アレイ・時間スケール。
- [P18] Wang et al., **Echo-state Networks Based on Voltage-controlled MTJs** (2026), https://doi.org/10.1103/qm64-x5g8 — VCMA-MTJの実機RC。
- [P19] Romera et al., **Vowel Recognition with Four Coupled Spin-torque Nano-oscillators** (2018), https://doi.org/10.1038/s41586-018-0632-y — 磁気振動子の実機調整・学習。
- [P22] Gartside et al., **Reconfigurable Training and Reservoir Computing in an Artificial Spin-vortex Ice** (2022), https://doi.org/10.1038/s41565-022-01091-7 — 再構成可能な磁気状態。
- [P23] Lee et al., **Task-adaptive Physical Reservoir Computing** (2024), https://doi.org/10.1038/s41563-023-01698-8 — 磁気相の選択とタスク適応。
- [P25] Torrejon et al., **Neuromorphic Computing with Nanoscale Spintronic Oscillators** (2017), https://doi.org/10.1038/nature23011 — スピン振動子RC。
- [P31] Namiki et al., **Magnetization Vector Rotation Reservoir Computing Operated by Redox Mechanism** (2024), https://doi.org/10.1021/acs.nanolett.3c05029 — 酸化還元による履歴。
- [P39] Grollier et al., **Neuromorphic Spintronics** (2020), https://doi.org/10.1038/s41928-019-0360-9 — スピントロニクスNeuromorphic総説。
- [P40] Allwood et al., **A Perspective on Physical Reservoir Computing with Nanomagnetic Devices** (2023), https://arxiv.org/abs/2212.04851 — ナノ磁石RCの展望。
- [A05] Ross et al., **Multilayer Spintronic Neural Networks with Radiofrequency Connections** (2023), https://doi.org/10.1038/s41565-023-01452-w — MTJをSynapse/Neuronとして使用。
- [A06] Meo et al., **Spin-wave Eigenmodes in Nanoscale MTJs with PMA** (2025), https://doi.org/10.1103/PhysRevApplied.23.034086 — MTJ内部のマグノンモード。
- [A07] Pham et al., **Skyrmion/MTJ電圧書込・読出し研究**（題名照合）, https://doi.org/10.1021/acs.nanolett.4c00316 — 持続的磁気記憶候補。
- [A08] **Current-controlled Nanomagnetic Writing**（題名照合）, https://doi.org/10.1038/s42005-020-00487-y — 状態書込の関連研究。

## D. スピン波・マグノン記憶・非線形計算

- [P02] Papp et al., **Nanoscale Neural Network Using Non-linear Spin-wave Interference** (2021), https://doi.org/10.1038/s41467-021-26711-z — YIG・内部磁場逆設計・SpinTorch。
- [P24] Voronov et al., **Inverse-design Topology Optimization of Magnonic Devices Using Level-set Method** (2025), https://doi.org/10.1038/s44306-025-00082-3 — 勾配・随伴法・NeuralMag。
- [P26] Nakane et al., **Reservoir Computing with Spin Waves Excited in a Garnet Film** (2018), https://doi.org/10.1109/ACCESS.2018.2794584 — ガーネットRC。
- [P27] Nakane et al., **Spin Waves Propagating through a Stripe Magnetic Domain Structure and Their Applications to RC** (2021), https://doi.org/10.1103/PhysRevResearch.3.033243 — 磁区と波動記憶。
- [P28] Nakane et al., **Performance Enhancement of a Spin-wave-based RC System Utilizing Different Physical Conditions** (2023), https://doi.org/10.1103/PhysRevApplied.19.034047 — 物理条件を変えた特徴集合。
- [P29] Namiki et al., **Experimental Demonstration of High-performance Physical Reservoir Computing with Nonlinear Interfered Spin Wave Multidetection** (2023), https://doi.org/10.1002/aisy.202300228 — 実YIG非線形スピン波RC。
- [P30] Namiki et al., **Fast Physical Reservoir Computing, Achieved with Nonlinear Interfered Spin Waves** (2024), https://doi.org/10.1088/2634-4386/ad561a — 高速スピン波RC。
- [P32] Vidamour et al., **Reconfigurable Reservoir Computing in a Magnetic Metamaterial** (2023), https://doi.org/10.1038/s42005-023-01352-4 — 磁気メタマテリアル。
- [A09] Namiki et al., **Iono–Magnonic Reservoir Computing with Chaotic Spin Wave Interference Manipulated by Ion-Gating** (2025), https://doi.org/10.1002/advs.202411777 — YIGのマグノン記憶・遅いゲート。
- [A10] Shinkai et al., **Distinct Memory Properties in Spin-wave Reservoir Computing Based on Synthetic Antiferromagnet** (2026, preprint), https://arxiv.org/abs/2603.22696 — SAFの異なる記憶モード。
- [A11] Breitbach et al., **All-magnonic Neurons for Analog Artificial Neural Networks**／題名・版の相違を照合 (2025/26, preprint), https://arxiv.org/abs/2509.18321 — 可変Fading Memory・カスケード。
- [A12] Abbott et al., **Perturbative Gradient Training: A Novel Training Paradigm for Bridging the Gap Between Deep Neural Networks and Physical Reservoir Computing** (2025, preprint), https://arxiv.org/abs/2506.04523 — YIGリングの実機と勾配近似。
- [A13] Rana & Otani, **Towards Magnonic Devices Based on Voltage-controlled Magnetic Anisotropy** (2019), https://doi.org/10.1038/s42005-019-0189-6 — VCMAとマグノン。
- [A14] Nizet et al., **Magnonic Memory／Computing関連研究**（題名照合）, https://doi.org/10.1063/5.0260884 — 磁気記憶・計算の展望。

## E. 磁性体・マグノニック素子の逆設計と自動探索

- [A15] Wang, Chumak & Pirro, **Inverse-design Magnonic Devices** (2021), https://doi.org/10.1038/s41467-021-22897-4 — YIGパターン・Direct Binary Search。
- [A16] Zenbaa et al., **A Universal Inverse-design Magnonic Device** (2025), https://doi.org/10.1038/s41928-024-01333-7 — 実YIG＋局所コイルを実測で最適化。
- [A17] Kiechle et al., **Experimental Demonstration of a Spin-Wave Lens Designed with Machine Learning** (2022), https://doi.org/10.1109/LMAG.2022.3209647 — SpinTorch→MuMax3→FIB製作→実測。
- [A18] Furuya et al., **Autonomous Synthesis System Integrating Theoretical, Informatics, and Experimental Approaches for Large-magnetic-anisotropy Materials** (2022), https://doi.org/10.1080/27660400.2022.2094698 — 異方性材料探索。
- [A19] **MTJトンネル障壁の原子配置を最適化する研究** (2023, 題名照合), https://doi.org/10.1103/PhysRevApplied.20.024044 — 量子・材料設計の隣接研究。
- [A20] Patel et al., **AI-guided Framework for the Design of Materials and Devices for MTJ-based True Random Number Generators** (2025), https://doi.org/10.1038/s44172-025-00376-8 — MTJ材料・物性のRL/進化探索。
- [A21] Nagaoka et al., **Inverse Design of Two-dimensional Magnonic Crystals via Topology Optimization with Frequency-domain Micromagnetics** (2026), https://doi.org/10.1002/sstr.70526 — マグノニック結晶探索。
- [A22] Vilsmeier et al., **Perspectives on Inverse Design for AI Magnonics** (2026, preprint), https://arxiv.org/abs/2607.07324 — 逆設計の展望。

## F. 微分可能磁気シミュレータの論文・手法

- [A23] Bruckner et al., **magnum.np関連論文**（正式題名照合）, https://doi.org/10.1038/s41598-023-39192-5 — PyTorchベースのLLG・逆問題。
- [A24] Abert et al., **NeuralMag関連論文**（正式題名照合）, https://doi.org/10.1038/s41524-025-01688-1 — Differentiable micromagnetics。

---

# 公開Repository・公式ソフト・データセット（このチャットで登場したもの）

**まず候補を調査してから、MVPで使用する最小限のツールだけ採用すること。** 公開されていることと「論文の全実験をそのまま再現できること」は異なる。依存ライブラリやCUDA互換性を必ず確認する。

### 磁気・物理シミュレーション／学習
- SpinTorch — https://github.com/a-papp/SpinTorch — PyTorchのスピン波逆設計、古いPyTorch/CUDA環境に注意。
- NeuralMag — https://gitlab.com/neuralmag/neuralmag — 微分可能micromagnetics、https://docs.neuralmag.org/。
- magnum.np — https://gitlab.com/magnum.np/magnum.np — PyTorch micromagnetics、https://magnum.np.gitlab.io/magnum.np/。
- MuMax3 — https://github.com/mumax/3 — GPU LLG、独立検証候補。
- MuMax⁺ — https://github.com/mumax/plus — Python API、多層磁性体候補。
- Hotspice — https://github.com/mumax/Hotspice — 熱活性ナノ磁石・spin ice。**コヒーレントな伝搬マグノンを扱うLLGの代用ではない。**
- Stochastic MTJ Model — https://github.com/utinclab/Stochastic_MTJ_Model — VCMA/STT/SOTを含むMTJモデル。YIG波動ソルバーの代用ではない。
- QuantumATK / COMSOL — 以前の会話で検討した隣接ソフト。MVPでは原則必須にしない（MTJ輸送／イオン輸送・多物理連成用）。

### Reservoir解析・Language Model・比較
- ReservoirPy — https://github.com/reservoirpy/reservoirpy — ESN/RCのベースライン。
- PRCpy — https://github.com/OSJL-py/PRCpy — 物理RCの実験データ・記憶指標解析。
- LAERC — https://github.com/fekoester/LAERC — Reservoir LMの学習・生成。
- Shakespeare_Res — https://github.com/fekoester/Shakespeare_Res — 文字予測のRC/AERC/Transformer。
- nanoGPT — https://github.com/karpathy/nanoGPT — 小規模Transformer baseline。
- Mamba — https://github.com/state-spaces/mamba — Selective State Space Model。
- S4 — https://github.com/state-spaces/s4 — 長期依存・Copying等の参考。
- Gated DeltaNet-2 — https://github.com/NVlabs/GatedDeltaNet-2 — 記憶の消去・書込分離の参考（実装する必要なし）。

### 逆設計・実験探索
- Optuna — https://github.com/optuna/optuna — 少数パラメータの試行管理。
- BoTorch — https://github.com/pytorch/botorch — Bayesian Optimization（公式移転などURLを確認）。
- Ax — https://github.com/facebook/Ax — 自動探索・実験管理。
- Nevergrad — https://github.com/facebookresearch/nevergrad — 勾配なし最適化。

### データセット
- Tiny Shakespeare — https://huggingface.co/datasets/karpathy/tiny_shakespeare
- TinyStories — https://huggingface.co/datasets/roneneldan/TinyStories
- WikiText-2 — 追加評価候補（公式配布場所を確認）。
- OpenWebText — LAERCの比較用。MVPでは原則不要。
- Selective Copying / Associative Recall — 再現可能なSeed・仕様・正解規則で合成生成し、生成コードも提供。

### もし入力環境に存在すれば参照する過去の試作資料（必須ではない）
- `sd11-x3ny.pdf`（LAERC論文原著）
- `spintronic_lm_research_bundle.zip`（過去の文献監査・概念設計・小モデル）
- `spintronic_physical_nn_mapping_124.xlsx`（材料・物理・NN対応表）
- `exp_aware_spintronics_bundle.zip`（以前の実験制約付きmacrospin試作）

これらのファイルが環境に存在しなければ、ないことを明記して公開文献から進めること。以前の試作の数値・コードも正しいとは前提しない。

---

# 最後の実行指示

**まず「文献とソフトの根拠確認 → 最小アーキテクチャ選択 → 物理ソルバー検証 → 内部Weight学習 → 固定Reservoir比較 → Tiny Shakespeare等で次token予測 → 2枚のA100による追加探索・評価 → 監査と結果報告」の順で、可能な限り実際に作って実行してください。**

- 学習が収束しない場合は、物理的に説明できる原因を切り分ける。任意のニューラルネットを足して「成功」にしない。
- 原著とコードの間に食い違いがあれば、それを説明し、修正または独自モデルとして明記する。
- **実行できた事実と、未実行・計画段階の作業を混同しない。**
- 最後に、私が全コードの理論と設計判断を読み解けるように、**各ファイルの役割、主要関数、論文の根拠、設定値の意味、失敗した条件、次に試す仮説**を日本語で丁寧に説明する。
