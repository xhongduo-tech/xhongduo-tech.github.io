---
pageClass: plain-doc
---

# Speech Technology

Speech technology is the body of techniques that lets machines hear, speak, and understand human speech, spanning the full pipeline from acoustic signal processing to end-to-end deep learning models and on to speech foundation models. This page organizes the complete topic list following classic textbooks such as *Speech Signal Processing* and *Spoken Language Processing*, along with mainstream course curricula.

## Topic Planning

<ProgressGrid cat="advanced/speech" />


### Fundamentals of Speech Production and Perception

- [ ] Speech organs and the mechanism of speech production: the source–filter model
- [ ] Acoustic classification of speech: vowels, consonants, and coarticulation
- [ ] Phonemes, syllables, and prosody: basic units of phonetics
- [ ] The human auditory system and auditory masking
- [ ] Critical bands, the Mel scale, and the Bark scale
- [ ] Loudness, pitch perception, and equal-loudness contours

### Digitization and Preprocessing of Speech Signals

- [ ] Sampling and quantization: the Nyquist theorem and aliasing
- [ ] Pre-emphasis and DC offset removal
- [ ] Framing and windowing: the short-time stationarity assumption and common window functions
- [ ] Short-time energy and short-time zero-crossing rate
- [ ] Endpoint detection (VAD): energy-based and model-based methods

### Spectral and Cepstral Analysis

- [ ] Discrete Fourier Transform (DFT) and Short-Time Fourier Transform (STFT)
- [ ] Generating and interpreting spectrograms
- [ ] Linear Predictive Coding (LPC) and LPC coefficients
- [ ] Cepstral analysis: homomorphic filtering and the complex cepstrum
- [ ] Pitch period estimation: the autocorrelation method and the YIN algorithm
- [ ] Formant estimation and vocal-tract parameter extraction

### Auditory Feature Extraction

- [ ] The complete Mel-Frequency Cepstral Coefficient (MFCC) pipeline
- [ ] Filter-bank features (FBank / log-Mel spectrogram)
- [ ] Perceptual Linear Prediction (PLP) coefficients
- [ ] First- and second-order differences: Delta and Delta-Delta features
- [ ] Extracting pitch and prosodic features
- [ ] Cepstral Mean and Variance Normalization (CMVN) and feature enhancement

### Hidden Markov Models and GMM-HMM Acoustic Models

- [ ] Markov chains and the three elements of Hidden Markov Models (HMMs)
- [ ] The three HMM problems: evaluation, decoding, and learning
- [ ] The forward–backward algorithm and the Viterbi algorithm
- [ ] The Baum-Welch algorithm and EM training
- [ ] Gaussian Mixture Models (GMMs) and continuous-density HMMs
- [ ] Triphone models and decision-tree state tying
- [ ] Forced alignment and pronunciation dictionaries
- [ ] Weighted Finite-State Transducer (WFST) decoding

### End-to-End Speech Recognition

- [ ] Connectionist Temporal Classification (CTC): alignment-free sequence modeling
- [ ] Attention-based encoder–decoder models (the LAS architecture)
- [ ] RNN-Transducer (RNN-T): a streaming recognition framework
- [ ] Hybrid multi-task learning with CTC and attention
- [ ] Conformer: convolution-augmented Transformer acoustic models
- [ ] Streaming recognition: chunk-based approaches and the U2/U2++ framework
- [ ] Whisper: large-scale weakly supervised multilingual recognition
- [ ] Language model fusion: shallow fusion, deep fusion, and cold fusion

### Speech Synthesis (TTS)

- [ ] Overview of speech synthesis systems: front-end text analysis and back-end acoustic modeling
- [ ] Text front end: tokenization, phonetic annotation, and prosody prediction
- [ ] Concatenative and parametric synthesis: a review of classical methods
- [ ] Tacotron / Tacotron 2: end-to-end sequence-to-sequence synthesis
- [ ] FastSpeech / FastSpeech 2: non-autoregressive fast synthesis and duration modeling
- [ ] VITS: end-to-end synthesis with variational inference and adversarial training
- [ ] Vocoder principles: from Griffin-Lim to neural vocoders
- [ ] WaveNet and WaveRNN: autoregressive waveform generation
- [ ] HiFi-GAN: generative adversarial network vocoders
- [ ] Diffusion vocoders: DiffWave and diffusion-based waveform generation
- [ ] Multi-speaker synthesis and voice cloning
- [ ] Emotion- and style-controllable speech synthesis

### Voice Conversion and Singing Voice Synthesis

- [ ] Voice Conversion: task definition and paradigms
- [ ] Voice conversion based on content–timbre disentanglement
- [ ] Zero-shot voice conversion and any-to-any conversion
- [ ] Singing Voice Synthesis (SVS): from sheet music to song, modeling pitch and rhythm
- [ ] Singing Voice Conversion and hands-on practice with so-vits-svc

### Speaker Recognition and Speaker Technology

- [ ] Speaker recognition tasks: verification and identification
- [ ] i-vectors and PLDA: classical speaker modeling
- [ ] x-vectors and deep speaker embeddings
- [ ] ECAPA-TDNN and time-delay neural network speaker models
- [ ] Loss functions for speaker embeddings: AAM-Softmax and metric learning
- [ ] Speaker diarization: who is speaking when

### Speech Enhancement and Noise Reduction

- [ ] Problem definition for speech enhancement: denoising, dereverberation, and echo cancellation
- [ ] Spectral subtraction and Wiener filtering: classical single-channel denoising
- [ ] Beamforming and microphone array signal processing
- [ ] Mask-based deep learning methods: IRM and PSM
- [ ] Acoustic Echo Cancellation (AEC) and double-talk detection
- [ ] Speech separation: Deep Clustering, Conv-TasNet, and target speaker extraction

### Keyword Spotting and Front-End Applications

- [ ] Keyword Spotting (KWS) system framework and false-alarm metrics
- [ ] HMM-based filler-word modeling for wake-up
- [ ] Wake-up models based on small deep networks
- [ ] Far-field speech interaction: front-end processing and recognition integration

### Speech Foundation Models and Audio Foundation Models

- [ ] Self-supervised speech pre-training: wav2vec 2.0 and contrastive learning
- [ ] HuBERT and WavLM: masked prediction for speech representation learning
- [ ] Neural audio codecs: SoundStream and EnCodec
- [ ] AudioLM and MusicLM: casting audio as a language model
- [ ] Discrete speech units (speech tokens) and speech language models
- [ ] GPT-4o voice mode and end-to-end speech dialogue architectures
- [ ] Full-duplex speech dialogue: simultaneous listening and speaking with interruption modeling

### Music Information Retrieval (MIR)

- [ ] Features of music signals: chroma features and beat
- [ ] Automatic Music Transcription (AMT) and multi-pitch estimation
- [ ] Music genre classification and tag prediction
- [ ] Audio fingerprinting and query-by-song
- [ ] Query by Humming
- [ ] Source separation: separating vocals from accompaniment

### Evaluation Metrics for Speech Systems

- [ ] Computing Word Error Rate (WER) and Character Error Rate (CER)
- [ ] Objective quality evaluation of speech: PESQ and STOI
- [ ] Subjective evaluation: MOS mean opinion score and ABX preference tests
- [ ] Speaker recognition evaluation: EER and minDCF
- [ ] Designing evaluation protocols for speech enhancement and synthesis

> After writing is complete: create a `xxx.md` file in this directory, then change the corresponding item above to `- [x] [Title](./xxx)`.
