---
pageClass: plain-doc
---

# AI Safety & Alignment

AI safety and alignment studies how to ensure that increasingly capable AI systems reliably serve human intentions and values. This post organizes all topics following the classic curriculum sequence of the alignment problem, interpretability, robustness, alignment techniques, evaluations & monitoring, and AI governance.

## Topic Plan

<ProgressGrid cat="advanced/ai-safety" />


### Part 1 The Alignment Problem

#### Chapter 1 The Alignment Problem in General
- [ ] What is the alignment problem: a conceptual framework for intent alignment and value learning
- [ ] Instrumental Convergence: why sufficiently intelligent systems compete for resources
- [ ] The Orthogonality Thesis: intelligence is independent of goals
- [ ] Case studies of Specification Gaming and Reward Hacking

#### Chapter 2 Outer Alignment and Inner Alignment
- [ ] Outer Alignment: whether reward functions can correctly express human intent
- [ ] Inner Alignment: goal-misaligned mesa-optimizers (Mesa-Optimizer)
- [ ] Deceptive Alignment: why models might fake alignment
- [ ] Goodhart's Law and its four forms in reward modeling

### Part 2 Interpretability

#### Chapter 3 Feature Visualization and Probing
- [ ] Neuron activation visualization and Feature Visualization
- [ ] Probing methods: linear probes and diagnosing internal representations
- [ ] Saliency maps and attribution methods: Integrated Gradients and Grad-CAM

#### Chapter 4 Mechanistic Interpretability
- [ ] Introduction to Mechanistic Interpretability: the Circuits perspective
- [ ] Transformer circuit analysis: Induction Heads and in-context learning
- [ ] Activation Patching and causal intervention methods
- [ ] The Superposition hypothesis and Polysemantic Neurons
- [ ] Sparse Autoencoders: extracting monosemantic features from superposition

### Part 3 Robustness

#### Chapter 5 Adversarial Examples
- [ ] The adversarial examples phenomenon and the linearity hypothesis
- [ ] Attack methods: FGSM, PGD, and the C&W attack
- [ ] Adversarial Training and Certified Robustness
- [ ] Universal adversarial suffixes on large language models

#### Chapter 6 Out-of-Distribution Generalization
- [ ] Distribution Shift and OOD Generalization
- [ ] OOD Detection: max-softmax, energy scores, and outlier exposure
- [ ] Spurious Correlations and Shortcut Learning

### Part 4 Red Teaming, Jailbreaks, and Agent Safety

#### Chapter 7 Red Teaming and Jailbreak Attacks
- [ ] Red Teaming methodology: from human red teams to automated red teams
- [ ] A taxonomy of jailbreaking attacks: role-playing, encoding bypass, and multi-turn elicitation
- [ ] Jailbreaking attacks and visual prompt injection on multimodal models

#### Chapter 8 Prompt Injection and Agent Safety
- [ ] Prompt Injection: direct and indirect injection
- [ ] Real-world threats of indirect prompt injection: malicious webpages, emails, and documents
- [ ] Agent safety: tool-call permissions, the principle of least privilege, and action confirmation mechanisms
- [ ] Designing agent Sandboxing and Capability Restriction

### Part 5 Hallucination and Factuality

#### Chapter 9 Hallucination
- [ ] Defining and classifying the hallucination phenomenon: intrinsic and extrinsic hallucination
- [ ] Causes of hallucination: training objectives, data noise, and knowledge boundaries
- [ ] Mitigation methods: Retrieval-Augmented Generation (RAG), citation generation, and uncertainty calibration
- [ ] Factuality evaluation benchmarks: TruthfulQA, FEVER, and fact-checking pipelines

### Part 6 Alignment Techniques

#### Chapter 10 Learning from Human Feedback
- [ ] The full RLHF pipeline: reward model training, PPO, and rejection sampling
- [ ] Reward model failure modes: overoptimization and reward tampering
- [ ] DPO and its variants: preference optimization without an explicit reward model
- [ ] Preference learning beyond RLHF: KTO, ORPO, and Process Reward Models (PRM)

#### Chapter 11 Constitutional AI and Scalable Oversight
- [ ] Constitutional AI: principle-driven self-critique and RLAIF
- [ ] The Scalable Oversight problem: how to supervise when humans cannot evaluate
- [ ] Debate and Recursive Reward Modeling
- [ ] Weak-to-Strong Generalization: can weak supervisors steer strong models

### Part 7 Evaluations & Monitoring

#### Chapter 12 Capability Evaluations
- [ ] Capability Evaluations methodology and the benchmark contamination problem
- [ ] Dangerous Capability Evals: biosecurity, cyber, and autonomous replication
- [ ] Frontier model safety frameworks: Responsible Scaling Policies (RSP) and evaluation trigger thresholds

#### Chapter 13 Alignment Evaluations and Deceptive Behavior
- [ ] Alignment Evals: measuring sycophancy and power-seeking tendencies
- [ ] Deceptive behavior evaluations: under what conditions models conceal their true intentions
- [ ] CoT Monitoring and its reliability limits
- [ ] Designing honeypot tests and Undercover Evaluations

### Part 8 AI Governance

#### Chapter 14 Regulatory Frameworks and International Coordination
- [ ] The EU AI Act: a risk-tiered regulatory framework
- [ ] US executive orders and voluntary commitments: the evolution of executive regulation
- [ ] China's generative AI measures and the algorithm registration system
- [ ] International agreements and coordination mechanisms: the AI Safety Summit and the International Scientific Report on the Safety of Advanced AI

#### Chapter 15 Compute Governance and Technical Governance Tools
- [ ] Compute Governance: chip export controls and training compute thresholds
- [ ] The governance trade-offs of open vs. closed model weights
- [ ] Structured Access and tiered API release

### Part 9 Long-Term Risk Discussion

#### Chapter 16 The AGI Risk Argument
- [ ] Tool AI vs. Agent AI: comparing the safety of two technical trajectories
- [ ] Core arguments for AGI risk: from alignment failure to loss-of-control scenarios
- [ ] Gradual vs. sudden risk: how capability emergence affects the safety timeline
- [ ] Principal critiques of, and rebuttals to, the AGI risk argument

### Part 10 Safety Engineering in Practice

#### Chapter 17 Content Moderation and Guardrail Systems
- [ ] Content Moderation pipelines: classifiers, moderation APIs, and human-in-the-loop workflows
- [ ] Guardrails: input filtering, output filtering, and constitutional classifiers
- [ ] Designing refusal mechanisms: over-refusal and refusal rate calibration

#### Chapter 18 Watermarking and Provenance
- [ ] Text Watermarking: statistical and embedded watermarks
- [ ] Tracing the provenance of generated images: the C2PA standard and content credentials
- [ ] Evaluating robustness attacks on watermarks and removal methods

> After writing: create `xxx.md` in this directory, then change the corresponding entry above to `- [x] [title](./xxx)`.
