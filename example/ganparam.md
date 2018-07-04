# Discriminator
- 入力 : MFCC24次元, 出力 : 値0-1の1次元
- loss function : CrossEntropyLoss
- loss はターゲット音声を1，Generatorで生成された音声を0とした2つの和 
- 2lossに重みなし
- optimizerはAdam, Rearning Lateは0.01
- 一回の学習につき全データ5週
# Generator
- 入力 : MFCC24次元, 出力 : MFCC24次元
- loss function : MSELoss, CrossEntropyLoss
- MSELossはGeneratorで生成された音声とターゲット音声とのロスに使用
- CrossEntropyLossは生成された音声がDiscriminatorに1と判断されたかとの誤差に使用
- 2lossに重みなし
- optimizerはAdam, Rearning Lateは0.01
- 一回の学習につき全データ10週
# others
- 学習の流れ : Discriminator学習後,Generator学習
- 学習の流れを全20週
- 音声データ216個中200を学習，16を評価に使用
- validationには200の1/10の20を使用
# Change points
- レGenerator WarmUp 追加
- Change training algorithm : G\_Warmup, for epoch(for minibatch(Dtrain, Gtrain))
- input change : MFCC24 -> MFCC24,DLT24,DLT224 -> add:before5,after5
