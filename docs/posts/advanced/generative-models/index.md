---
pageClass: plain-doc
---

# 生成模型

Goodfellow《深度学习》第20章、扩散模型综述。按照「学完一个学科 = 写完该学科权威教材对应的全部博文」的标准，每写完一篇勾掉一条。

## 主题规划

<ProgressGrid cat="advanced/generative-models" />

### 第一篇 生成模型基础

- [x] [概率生成模型框架](./probabilistic-generative-model-framework)
- [x] [最大似然与隐变量模型](./maximum-likelihood-latent-variable)
- [x] [自回归模型（PixelRNN/Transformer）](./autoregressive-models)
- [x] [变分自编码器（VAE）](./variational-autoencoder)
- [x] [VAE 的改进（β-VAE/VQ-VAE）](./vae-improvements)
- [x] [归一化流模型](./normalizing-flows)

### 第二篇 对抗生成

- [x] [生成对抗网络（GAN）](./generative-adversarial-networks)
- [x] [GAN 的训练动力学与模式坍缩](./gan-training-dynamics-mode-collapse)
- [x] [GAN 的改进（WGAN/SNGAN）](./gan-improvements-wgan-sngan)
- [x] [条件 GAN](./conditional-gan)
- [x] [StyleGAN 与图像生成](./stylegan-image-generation)
- [x] [GAN 与图像翻译（Pix2Pix/CycleGAN）](./gan-image-translation)

### 第三篇 扩散模型

- [x] [去噪扩散模型（DDPM）](./denoising-diffusion-probabilistic-models)
- [x] [扩散模型的采样与加速](./diffusion-sampling-acceleration)
- [x] [潜在扩散模型（LDM）](./latent-diffusion-models)
- [x] [基于分数的生成模型](./score-based-generative-models)
- [x] [一致性模型与流匹配](./consistency-models-flow-matching)
- [x] [条件扩散（ControlNet）](./conditional-diffusion-controlnet)
- [x] [文生图（Stable Diffusion）](./text-to-image-stable-diffusion)

### 第四篇 大模型生成与前沿

- [x] [文本生成与大语言模型衔接](./text-generation-llm)
- [x] [视频生成模型](./video-generation)
- [x] [3D 生成](./3d-generation)
- [x] [音乐与语音生成](./music-speech-generation)
- [x] [多模态生成](./multimodal-generation)
- [x] [生成模型的可控性](./generative-model-controllability)
- [x] [生成模型的安全与版权](./generative-model-safety-copyright)
- [x] [生成模型的评估](./generative-model-evaluation)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
