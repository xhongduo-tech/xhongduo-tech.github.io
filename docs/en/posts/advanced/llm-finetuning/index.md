---
pageClass: plain-doc
---

# LLM Fine-tuning

Adapting general-purpose models to specific tasks and values: a complete knowledge tree spanning continued pre-training, instruction tuning, and preference alignment.

## Topic Planning

<ProgressGrid cat="advanced/llm-finetuning" />


### Part 1 — Fine-tuning Paradigms

- [ ] Overview of fine-tuning paradigms: the three stages from pre-training and instruction tuning to alignment
- [ ] Continual Pre-Training: corpus selection, mixture ratios, and training strategies
- [ ] The trade-off between domain-adaptive pre-training and preserving general capabilities
- [ ] Supervised Fine-Tuning (SFT): task definition and training objective of instruction tuning
- [ ] Instruction generalization: task diversity, scaling effects, and instruction evolution (Evol-Instruct)
- [ ] Overview of alignment paradigms: the two main routes of RLHF and direct preference optimization
- [ ] Scaling laws for fine-tuning: data volume, model scale, and downstream returns
- [ ] Small-sample efficient fine-tuning: LIMA's "Superficial Alignment Hypothesis" and the debate around it

### Part 2 — Data Engineering

- [ ] Three major sources of instruction data: human annotation, model distillation, and self-generation
- [ ] Self-Instruct: automated instruction generation pipelines and filtering strategies
- [ ] WizardLM's instruction evolution: deep evolution and wide evolution
- [ ] Data quality filtering: perplexity, IFD (Instruction-Following Difficulty), and de-duplication/de-contamination
- [ ] Data diversity and mixture: task mixing, length distribution, and sampling curricula
- [ ] Construction, organization, and quality control of multi-turn dialogue data
- [ ] Chat templates: ChatML, the Llama family of templates, and special token design
- [ ] Loss masking: implementation details and common mistakes in computing loss only over answer tokens
- [ ] Packing: sample concatenation, cross-sample attention isolation, and position encoding handling
- [ ] Dataset formats and tooling: converting and validating between the Alpaca and ShareGPT formats

### Part 3 — Full-Parameter Fine-Tuning

- [ ] Memory ledger: precise accounting of parameters, gradients, optimizer states, and activations
- [ ] Mixed-precision training: numerical stability of BF16/FP32 and the memory cost of AdamW
- [ ] Gradient checkpointing: activation recomputation that trades compute for memory
- [ ] DeepSpeed ZeRO: ZeRO-1/2/3 sharding of optimizer states, gradients, and parameters
- [ ] FSDP: the principles and configuration of PyTorch's native fully-sharded data parallelism
- [ ] The trade-offs of tensor parallelism versus pipeline parallelism in fine-tuning scenarios
- [ ] Offloading techniques: CPU/NVMe offload and the paged optimizer
- [ ] Training throughput tuning: MFU analysis from a single GPU to multiple GPUs
- [ ] Training stability: loss spikes, gradient clipping, and learning-rate scheduling

### Part 4 — Parameter-Efficient Fine-Tuning (PEFT)

- [ ] PEFT overview: additive, selective, and reparameterization methods
- [ ] Adapters: bottleneck structure, placement, and their variants
- [ ] Prefix-Tuning and Prompt-Tuning: learnable continuous prompts
- [ ] P-Tuning v2: deep prompt tuning for models of all sizes
- [ ] LoRA principles: the low-rank assumption, rank selection, and target modules
- [ ] LoRA engineering details: initialization, scaling factor, and weight merging at inference time
- [ ] QLoRA: the synergy of NF4 quantization, double quantization, and the paged optimizer
- [ ] DoRA: direction-magnitude decoupled fine-tuning via weight decomposition
- [ ] PiSSA: principal singular component initialization and its convergence-acceleration mechanism
- [ ] LoRA+ and rsLoRA: asymmetric learning rates and rank-stabilized scaling
- [ ] AdaLoRA: adaptive rank allocation based on importance scores
- [ ] Modular composition of LoRA: multi-LoRA fusion and MoE-style PEFT

### Part 5 — Long-Context Fine-Tuning

- [ ] Challenges of long-context fine-tuning: attention memory, position extrapolation, and data scarcity
- [ ] Position encoding extension: positional interpolation (PI), NTK, and YaRN
- [ ] Sequence parallelism: attention-head splitting in DeepSpeed Ulysses
- [ ] Ring Attention: ring communication and load balancing for blockwise attention
- [ ] Constructing long-sequence training data: long-document continuation, long dialogues, and synthetic long-range dependency tasks
- [ ] Long-context capability evaluation: Needle-in-a-Haystack (NIAH) and the RULER benchmark

### Part 6 — Preference Optimization

- [ ] RLHF end-to-end overview: the three stages of SFT → reward model → reinforcement learning
- [ ] Preference data collection: annotation protocols, consistency checks, and the Bradley-Terry model
- [ ] Reward model training: ranking loss, model ensembling, and reward calibration
- [ ] Implementing PPO for LLMs: KL penalty, the value network, and engineering tuning tips
- [ ] Rejection sampling fine-tuning: Best-of-N distillation, RFT, and RAFT
- [ ] ReST and iterative rejection sampling: self-improvement in the data–training loop
- [ ] DPO derivation: the full mathematical chain from the RLHF objective to a closed-form optimal solution
- [ ] DPO in practice: reference models, the temperature coefficient β, and common training pitfalls
- [ ] IPO and KTO: unpaired preference data and the introduction of prospect theory
- [ ] ORPO and SimPO: direct preference optimization without a reference model
- [ ] GRPO: group-relative advantage estimation and its simplification of PPO
- [ ] RLVR: reinforcement learning with verifiable rewards (mathematical and code reasoning)
- [ ] RLAIF and Constitutional AI: replacing human annotation with AI feedback
- [ ] Process reward models (PRM): stepwise supervision and reasoning-chain alignment
- [ ] Trade-offs between online and offline preference optimization: iterative DPO and the distribution shift problem

### Part 7 — Domain and Multimodal Fine-Tuning

- [ ] Domain fine-tuning methodology: corpus mixture, capability preservation, and evaluation baselines
- [ ] Medical LLM fine-tuning: corpus sources, privacy compliance, and hallucination control
- [ ] Legal and financial domain fine-tuning: terminology alignment and factual constraints
- [ ] Code model fine-tuning: data recipes and Fill-in-the-Middle for CodeLlama/StarCoder
- [ ] Mathematical reasoning fine-tuning: synthetic problem data, verifiers, and rejection sampling
- [ ] LLaVA-style visual instruction tuning: vision-language alignment pre-training and instruction data construction
- [ ] Multimodal dialogue data: visual grounding, OCR, and interleaved image-text data

### Part 8 — Framework Practice

- [ ] Choosing a fine-tuning framework: a head-to-head comparison of LLaMA-Factory, Axolotl, ms-swift, and TRL
- [ ] LLaMA-Factory in practice: configuration files, WebUI, and the full LoRA/QLoRA workflow
- [ ] Axolotl in practice: declarative YAML configuration and multi-GPU training
- [ ] ms-swift in practice: adapting domestic models and bridging to lightweight deployment
- [ ] TRL in practice: the programming interfaces of SFTTrainer, DPOTrainer, and PPOTrainer
- [ ] Productionizing the training pipeline: data validation, checkpoint resumption, and experiment tracking

### Part 9 — Evaluation and Diagnosis

- [ ] A framework for evaluating fine-tuning outcomes: the three dimensions of capability, alignment, and safety
- [ ] Using general capability benchmarks correctly: evaluation details of MMLU, C-Eval, and GSM8K
- [ ] Instruction-following evaluation: IFEval and FollowBench
- [ ] Alignment quality evaluation: bias analysis of MT-Bench, AlpacaEval, and LLM-as-Judge
- [ ] Catastrophic forgetting: measurement methods and mitigations such as replay, regularization, and module isolation
- [ ] Overfitting and memorization: detecting training-set leakage and verbatim regurgitation
- [ ] Reward hacking: identifying, measuring, and mitigating reward model misspecification
- [ ] Confounding factors in evaluation: length bias, format bias, and sampling randomness
- [ ] Safety and jailbreak robustness evaluation

> Note: vLLM and SGLang are inference engines, so they belong under the [LLM Deployment](/en/posts/advanced/llm-deployment/) module.

> After writing: create a new `xxx.md` in this directory, then change the corresponding item above to `- [x] [Title](./xxx)`.
