---
pageClass: plain-doc
---

# 语音 · ASR 与 TTS

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Lawrence R. Rabiner & Ronald W. Schafer, "Theory and Applications of Digital Speech Processing" (2011)
- Xuedong Huang, Alex Acero & Hsiao-Wuen Hon, "Spoken Language Processing" (2001)
- Dong Yu & Li Deng, "Automatic Speech Recognition: A Deep Learning Approach" (2015)

## 主题规划

<ProgressGrid cat="advanced/asr-tts" />

### 第1篇

- [x] [语音前端与端点检测（VAD） (Rabiner §9)](./speech-frontend-vad)
- [x] [声学特征 MFCC/Filterbank (Rabiner §5)](./acoustic-features-mfcc)
- [x] [隐马尔可夫模型 HMM (Huang §8)](./hidden-markov-model)
- [x] [CTC 损失与端到端 (Yu & Deng §7)](./acoustic-model-dnn-tdnn)
- [x] [声学模型 DNN/TDNN (Yu & Deng §6)](./conformer)
- [x] [语言模型与解码 (Huang §13)](./language-model-decoding)
- [x] [RNN-T（RNN Transducer） (Graves, RNN-T 2012)](./rnn-transducer)
- [x] [端到端 ASR Conformer (Gulati et al., Conformer 2020)](./ctc-loss-end-to-end)

### 第2篇

- [x] [大规模 ASR（Whisper） (Radford et al., Whisper 2022)](./hidden-markov-model)
- [x] [Tacotron 端到端 TTS (Shen et al., Tacotron 2018)](./language-model-decoding)
- [x] [声码器 WaveNet/HiFi-GAN (Oord et al., WaveNet 2016; Kong et al., HiFi-GAN 2020)](./rnn-transducer)
- [x] [语音增强与分离 (Huang §15)](./speech-enhancement-separation)
- [x] [识别评测指标（WER/CER 与对齐） (Yu & Deng §3)](./speech-frontend-vad)
