# REFERENCES - 確認範囲を区別した文献一覧

2026-10-14に原典URLとCrossref/arXivへ照合。書誌情報の一致は本文の全検証を意味しない。全67件の完全再現は行っていない。

## 主に使った根拠

- **P01 Reservoir computing as a language model**: [https://doi.org/10.1103/sd11-x3ny](https://doi.org/10.1103/sd11-x3ny)。原著preprint本文 §III.1/§V/§VI、掲載書誌・公式repo確認。線形RC readout/文字LM評価を参考。
- **P02 Nanoscale neural network using non-linear spin-wave interference**: [https://doi.org/10.1038/s41467-021-26711-z](https://doi.org/10.1038/s41467-021-26711-z)。原著preprint全文・Methods §5.1-5.3・公式solver確認。LLG/場学習の考え方の根拠。独自縮約を再現と称さない。
- **A16 A universal inverse-design magnonic device**: [https://doi.org/10.1038/s41928-024-01333-7](https://doi.org/10.1038/s41928-024-01333-7)。原著preprint本文・Methods §III-A・Fig.1確認。YIGの局所DCコイル更新の根拠。掲載版の補足PDF取得はDNS失敗。

追加の分散式はLLG+uniform厚みの磁静力学から独自導出した。Kalinikos & Slavin (1986), DOI [10.1088/0022-3719/19/35/014](https://doi.org/10.1088/0022-3719/19/35/014) は参考文献だが本文アクセス制限のため式番号まで未確認。

PyTorch 2.5.1の公式 [complex autograd文書](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/complex_numbers.rst) と実動作でFFT/複素勾配を確認。

## 全候補の書誌確認

| ID | 確認した正式題名 | URL | 確認範囲 |
|---|---|---|---|
| P01 | Reservoir computing as a language model | [原典](https://doi.org/10.1103/sd11-x3ny) | 選定原著本文確認（版は上記） |
| P33 | Harnessing Nonlinearity: Predicting Chaotic Systems and Saving Energy in Wireless Communication | [原典](https://doi.org/10.1126/science.1091277) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P34 | Recent advances in physical reservoir computing: A review | [原典](https://doi.org/10.1016/j.neunet.2019.03.005) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P35 | In-sensor reservoir computing for language learning via two-dimensional memristors | [原典](https://doi.org/10.1126/sciadv.abg1455) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P36 | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | [原典](https://arxiv.org/abs/2312.00752) | arXiv abstract page retrieved / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P37 | RWKV: Reinventing RNNs for the Transformer Era | [原典](https://doi.org/10.18653/v1/2023.findings-emnlp.936) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P38 | Attention-enhanced reservoir computing | [原典](https://doi.org/10.1103/PhysRevApplied.22.014039) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P41 | Physical reservoir computing with emerging electronics | [原典](https://doi.org/10.1038/s41928-024-01133-z) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P42 | Topological magnetic and ferroelectric systems for reservoir computing | [原典](https://doi.org/10.1038/s42254-024-00729-w) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P43 | Metrics for spin-based computing | [原典](https://doi.org/10.1038/s42254-025-00918-1) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| A01 | Information Processing Capacity of Dynamical Systems | [原典](https://doi.org/10.1038/srep00514) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| A02 | Attention Is All You Need | [原典](https://arxiv.org/abs/1706.03762) | arXiv abstract page retrieved / 本文・実験種別・適用項は未精査、実装根拠にしない |
| A03 | Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention | [原典](https://arxiv.org/abs/2605.22791) | arXiv abstract page retrieved / 本文・実験種別・適用項は未精査、実装根拠にしない |
| A04 | TinyStories: How Small Can Language Models Be and Still Speak Coherent English? | [原典](https://arxiv.org/abs/2305.07759) | arXiv abstract page retrieved / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P04 | Deep reservoir computing: A critical experimental analysis | [原典](https://doi.org/10.1016/j.neucom.2016.12.089) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P05 | Multilayer Reservoir Computing Based on Ferroelectric α‐In
                    <sub>2</sub>
                    Se
                    <sub>3</sub>
                    for Hierarchical Information Processing | [原典](https://doi.org/10.1002/adma.202108826) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P06 | Deep photonic reservoir computer based on frequency multiplexing with fully analog connection between layers | [原典](https://doi.org/10.1364/OPTICA.489501) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P07 | Deep photonic reservoir computing recurrent network | [原典](https://doi.org/10.1364/OPTICA.506635) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P08 | A high-performance deep reservoir computer experimentally demonstrated with ion-gating reservoirs | [原典](https://doi.org/10.1038/s44172-024-00227-y) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P09 | Neuromorphic overparameterisation and few-shot learning in multilayer physical neural networks | [原典](https://doi.org/10.1038/s41467-024-50633-1) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P20 | Deep physical neural networks trained with backpropagation | [原典](https://doi.org/10.1038/s41586-021-04223-6) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P21 | Physical deep learning with biologically inspired training method: gradient-free approach for physical hardware | [原典](https://doi.org/10.1038/s41467-022-35216-2) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P03 | Trainable neuromorphic spintronic hardware Via analog finite-difference gradient methods | [原典](https://doi.org/10.1038/s41467-026-73624-w) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P10 | Reservoir computing with dipole-coupled nanomagnets | [原典](https://doi.org/10.7567/1347-4065/ab2406) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P11 | Spintronic reservoir computing without driving current or magnetic field | [原典](https://doi.org/10.1038/s41598-022-14738-1) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P12 | Passive frustrated nanomagnet reservoir computing | [原典](https://doi.org/10.1038/s42005-023-01324-8) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P13 | Radio-frequency-modulated artificial synapses based on magnetic tunnel junctions with perpendicular magnetic anisotropy | [原典](https://doi.org/10.1103/PhysRevApplied.21.014020) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P14 | An Antiferromagnetic Neuromorphic Memory Based on Perpendicularly Magnetized CoO | [原典](https://doi.org/10.1021/acs.nanolett.4c02340) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P15 | Magneto-ionic synapse for reservoir computing | [原典](https://doi.org/10.1103/PhysRevApplied.23.054043) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P16 | Magneto-Ionic Physical Reservoir Computing in Perpendicularly Magnetized Heterostructures | [原典](https://doi.org/10.1021/acs.nanolett.5c03889) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P17 | Perpendicular-anisotropy artificial spin ice with spontaneous ordering: a platform for reservoir computing with flexible timescales | [原典](https://doi.org/10.1038/s44172-025-00499-y) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P18 | Echo-state networks based on voltage-controlled magnetic tunnel junctions | [原典](https://doi.org/10.1103/qm64-x5g8) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P19 | Vowel recognition with four coupled spin-torque nano-oscillators | [原典](https://doi.org/10.1038/s41586-018-0632-y) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P22 | Reconfigurable training and reservoir computing in an artificial spin-vortex ice via spin-wave fingerprinting | [原典](https://doi.org/10.1038/s41565-022-01091-7) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P23 | Task-adaptive physical reservoir computing | [原典](https://doi.org/10.1038/s41563-023-01698-8) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P25 | Neuromorphic computing with nanoscale spintronic oscillators | [原典](https://doi.org/10.1038/nature23011) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P31 | Magnetization Vector Rotation Reservoir Computing Operated by Redox Mechanism | [原典](https://doi.org/10.1021/acs.nanolett.3c05029) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P39 | Neuromorphic spintronics | [原典](https://doi.org/10.1038/s41928-019-0360-9) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P40 | A perspective on physical reservoir computing with nanomagnetic devices | [原典](https://arxiv.org/abs/2212.04851) | arXiv abstract page retrieved / 本文・実験種別・適用項は未精査、実装根拠にしない |
| A05 | Multilayer spintronic neural networks with radiofrequency connections | [原典](https://doi.org/10.1038/s41565-023-01452-w) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| A06 | Spin-wave eigenmodes in nanoscale magnetic tunnel junctions with perpendicular magnetic anisotropy | [原典](https://doi.org/10.1103/PhysRevApplied.23.034086) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| A07 | Electrical Detection and Nucleation of a Magnetic Skyrmion in a Magnetic Tunnel Junction Observed via <i>Operando</i> Magnetic Microscopy | [原典](https://doi.org/10.1021/acs.nanolett.4c00316) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| A08 | Current-controlled nanomagnetic writing for reconfigurable magnonic crystals | [原典](https://doi.org/10.1038/s42005-020-00487-y) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P02 | Nanoscale neural network using non-linear spin-wave interference | [原典](https://doi.org/10.1038/s41467-021-26711-z) | 選定原著本文確認（版は上記） |
| P24 | Inverse-design topology optimization of magnonic devices using level-set method | [原典](https://doi.org/10.1038/s44306-025-00082-3) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P26 | Reservoir Computing With Spin Waves Excited in a Garnet Film | [原典](https://doi.org/10.1109/ACCESS.2018.2794584) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P27 | Spin waves propagating through a stripe magnetic domain structure and their applications to reservoir computing | [原典](https://doi.org/10.1103/PhysRevResearch.3.033243) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P28 | Performance Enhancement of a Spin-Wave-Based Reservoir Computing System Utilizing Different Physical Conditions | [原典](https://doi.org/10.1103/PhysRevApplied.19.034047) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P29 | Experimental Demonstration of High‐Performance Physical Reservoir Computing with Nonlinear Interfered Spin Wave Multidetection | [原典](https://doi.org/10.1002/aisy.202300228) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P30 | Fast physical reservoir computing, achieved with nonlinear interfered spin waves | [原典](https://doi.org/10.1088/2634-4386/ad561a) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| P32 | Reconfigurable reservoir computing in a magnetic metamaterial | [原典](https://doi.org/10.1038/s42005-023-01352-4) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| A09 | Iono–Magnonic Reservoir Computing With Chaotic Spin Wave Interference Manipulated by Ion‐Gating | [原典](https://doi.org/10.1002/advs.202411777) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| A10 | Distinct memory properties in spin-wave reservoir computing based on synthetic antiferromagnet | [原典](https://arxiv.org/abs/2603.22696) | arXiv abstract page retrieved / 本文・実験種別・適用項は未精査、実装根拠にしない |
| A11 | An all-magnonic neuron with tunable fading memory | [原典](https://arxiv.org/abs/2509.18321) | arXiv abstract page retrieved / 本文・実験種別・適用項は未精査、実装根拠にしない |
| A12 | Perturbative Gradient Training: A novel training paradigm for bridging the gap between deep neural networks and physical reservoir computing | [原典](https://arxiv.org/abs/2506.04523) | arXiv abstract page retrieved / 本文・実験種別・適用項は未精査、実装根拠にしない |
| A13 | Towards magnonic devices based on voltage-controlled magnetic anisotropy | [原典](https://doi.org/10.1038/s42005-019-0189-6) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| A14 | Perspective on nonvolatile magnon-signal storage and in-memory computation for low-power consuming magnonics | [原典](https://doi.org/10.1063/5.0260884) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| A15 | Inverse-design magnonic devices | [原典](https://doi.org/10.1038/s41467-021-22897-4) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| A16 | A universal inverse-design magnonic device | [原典](https://doi.org/10.1038/s41928-024-01333-7) | 選定原著本文確認（版は上記） |
| A17 | Experimental Demonstration of a Spin-Wave Lens Designed With Machine Learning | [原典](https://doi.org/10.1109/LMAG.2022.3209647) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| A18 | Autonomous synthesis system integrating theoretical, informatics, and experimental approaches for large-magnetic-anisotropy materials | [原典](https://doi.org/10.1080/27660400.2022.2094698) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| A19 | Quantum Annealing Optimization Method for the Design of Barrier Materials in Magnetic Tunnel Junctions | [原典](https://doi.org/10.1103/PhysRevApplied.20.024044) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| A20 | AI-guided framework for the design of materials and devices for magnetic-tunnel-junction-based true random number generators | [原典](https://doi.org/10.1038/s44172-025-00376-8) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| A21 | Inverse Design of Two‐Dimensional Magnonic Crystals via Topology Optimization with Frequency‐Domain Micromagnetics | [原典](https://doi.org/10.1002/sstr.70526) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| A22 | Perspectives on inverse design for AI magnonics | [原典](https://arxiv.org/abs/2607.07324) | arXiv abstract page retrieved / 本文・実験種別・適用項は未精査、実装根拠にしない |
| A23 | magnum.np: a PyTorch based GPU enhanced finite difference micromagnetic simulation framework for high level development and inverse design | [原典](https://doi.org/10.1038/s41598-023-39192-5) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |
| A24 | NeuralMag: an open-source nodal finite-difference code for inverse micromagnetics | [原典](https://doi.org/10.1038/s41524-025-01688-1) | metadata verified / 本文・実験種別・適用項は未精査、実装根拠にしない |

正確な著者・刊行日・DOI・登録licenseは `reference_registry.json` に保存。Crossref licenseは記事の権利でありコードlicenseとは別。

題名が曖昧だったA07/A08/A14/A19/A23/A24はregistryの正式題名に修正。A11は取得したarXivでは `An all-magnonic neuron with tunable fading memory`。BoTorchの公式repoはmeta-pytorchへ移転している。

## ソフトウェア確認

| ソフト | 正式repo | 公開license情報 | 利用 |
|---|---|---|---|
| GatedDeltaNet-2 | [https://github.com/NVlabs/GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2) | NOASSERTION | 調査のみ、導入しない |
| PRCpy | [https://github.com/OSJL-py/PRCpy](https://github.com/OSJL-py/PRCpy) | MIT | 調査のみ、導入しない |
| SpinTorch | [https://github.com/a-papp/SpinTorch](https://github.com/a-papp/SpinTorch) | 未判定 | 調査のみ、導入しない |
| Ax | [https://github.com/facebook/Ax](https://github.com/facebook/Ax) | MIT | 調査のみ、導入しない |
| nevergrad | [https://github.com/facebookresearch/nevergrad](https://github.com/facebookresearch/nevergrad) | MIT | 調査のみ、導入しない |
| LAERC | [https://github.com/fekoester/LAERC](https://github.com/fekoester/LAERC) | 未判定 | 調査のみ、導入しない |
| Shakespeare_Res | [https://github.com/fekoester/Shakespeare_Res](https://github.com/fekoester/Shakespeare_Res) | NOASSERTION | 調査のみ、導入しない |
| nanoGPT | [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | MIT | 調査のみ、導入しない |
| 3 | [https://github.com/mumax/3](https://github.com/mumax/3) | NOASSERTION | 調査のみ、導入しない |
| Hotspice | [https://github.com/mumax/Hotspice](https://github.com/mumax/Hotspice) | GPL-3.0 | 調査のみ、導入しない |
| plus | [https://github.com/mumax/plus](https://github.com/mumax/plus) | GPL-3.0 | 調査のみ、導入しない |
| optuna | [https://github.com/optuna/optuna](https://github.com/optuna/optuna) | MIT | 調査のみ、導入しない |
| botorch | [https://github.com/meta-pytorch/botorch](https://github.com/meta-pytorch/botorch) | MIT | 調査のみ、導入しない |
| reservoirpy | [https://github.com/reservoirpy/reservoirpy](https://github.com/reservoirpy/reservoirpy) | MIT | 調査のみ、導入しない |
| mamba | [https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba) | Apache-2.0 | 調査のみ、導入しない |
| s4 | [https://github.com/state-spaces/s4](https://github.com/state-spaces/s4) | Apache-2.0 | 調査のみ、導入しない |
| Stochastic_MTJ_Model | [https://github.com/utinclab/Stochastic_MTJ_Model](https://github.com/utinclab/Stochastic_MTJ_Model) | MIT | 調査のみ、導入しない |

SpinTorchには `licenses/LICENSE_Wavetorch.md` があるが、APIが全repoのlicenseを判定していない。今回solverのソースをコピーして利用せず独自実装した。LAERCのlicenseも自動判定不能で、導入していない。

[NeuralMag公式docs](https://docs.neuralmag.org/) はMIT、JAX/PyTorch、LLG/交換/demag等を確認。magnum.np公式docsは取得時502で、互換性試験は行っていない。MuMax3などの候補も実行互換性を実証していない。

## データと取得失敗

Tiny ShakespeareはKarpathyの[原配布](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare)を取得しSHA-256を記録した。HF cardのlicense欄は `More Information Needed` で、明示licenseを確認できない。原文はShakespeareのpublic-domain作品だが、配布ファイルの再配布権を勝手に断定しない。

掲載Natureページはブラウザ取得で認証redirect、Papp本文はarXiv PDFを取得して全文抽出、ZenbaaはarXiv HTMLを確認した。Zenbaa掲載補足PDFはDNS失敗。過去の試作bundle/pdf/xlsxはこの作業ディレクトリに存在せず、採用していない。
