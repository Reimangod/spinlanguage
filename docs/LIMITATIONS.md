# 主張できる範囲

- 自作の縮約モデル内の実計算。独立MuMax/NeuralMag検証・実測校正は未実行。
- 周期1D・最低厚みモード・弱非線形の近似。全反磁界の非線形や強いmagnon scatteringは省略。
- Ms、A、alphaは代表値。実ウェハ物性・温度・バラつきは不明。
- コイル応答、RF/receiver帯域、5 microtesla入力、絶対電圧ゲインは仮定。装置全体の消費エネルギーは未算定。
- 一窓32文字の訓練。低減衰だけを根拠に長文の意味記憶を主張しない。自由生成のstate carryは訓練より長い条件になる。
- Delayed Recallは1個の8値symbolであり、複数key-valueのAssociative RecallやSelective Copyingを再現していない。
- Tiny Shakespeareの連続文字分割は同じ作者・作品群を共有する。別作品/別作者への汎化は未確認。
- 3 Seedは小さな再現性検査。小効果の統計的確立にはSeed追加と別コーパスが必要。
- GRUは同じ更新数・データで比較するが、モデルパラメータ数は完全一致していない。
- ノイズ/DAC/帯域アブレーションは原則学習済み重みを変更して評価する感度試験。ノイズ下の再学習最適性能ではない。
- 外側探索はポート配置2候補のみ。validation選択であり、全構造空間の最適解ではない。
- `last.pt` はoptimizer/RNGを含むが、自動再開CLIは初版では用意していない。未知のjob状態で自動再投入しない。
- 共有保存先満杯のためRAM領域で計算し、完成runごとにMacへ保存。進行中runの最後のcheckpointは転送前のSSH障害に弱く、再開前には保存証拠の照合が必要。
- Tiny ShakespeareのHF cardはlicenseを明示していない。Shakespeare原文のpublic-domain性と、配布packageの権利/annotationは区別する。研究私用の取得・provenanceを記録し、再配布licenseを勝手にCC0としない。
- 元の文献一覧の全67件の本文/補足/コードは読み切ったと主張しない。本文を確認した選定根拠と書誌確認のみを分ける。
