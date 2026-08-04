---
pageClass: plain-doc
---

# Deep Learning

Aligned with the chapter structure of Goodfellow's *Deep Learning* (the "Flowers book") and Mu Li's *Dive into Deep Learning*, covering a complete learning path from mathematical foundations to engineering practice. Finishing a classic textbook = writing the blog post for every one of its chapters.

## Topic Roadmap

<ProgressGrid cat="advanced/deep-learning" />


### Part I Math Foundations Review

- [ ] Linear algebra: scalars, vectors, matrices, and tensors
- [ ] Matrix decomposition: eigendecomposition and singular value decomposition
- [ ] Norms, trace operations, and the pseudoinverse
- [ ] Probability basics: random variables and probability distributions
- [ ] Common distributions: Bernoulli, Gaussian, and the exponential family
- [ ] Conditional probability, expectation, variance, and covariance
- [ ] Maximum likelihood estimation and Bayesian statistics
- [ ] Information theory: entropy, cross-entropy, and KL divergence
- [ ] Numerical computation: overflow, underflow, and ill-conditioning
- [ ] Gradients, Jacobian matrices, and Hessian matrices
- [ ] Calculus and the basics of automatic differentiation

### Part II Deep Learning Fundamentals

- [ ] Machine learning basics: capacity, overfitting, and underfitting
- [ ] The no-free-lunch theorem and the idea of regularization
- [ ] Hyperparameters, validation sets, and cross-validation
- [ ] Linear regression and implementing it from scratch
- [ ] Softmax regression and classification problems
- [ ] Perceptrons and multi-layer perceptrons
- [ ] Deep feedforward networks: hidden units and the universal approximation theorem
- [ ] Activation functions: Sigmoid, Tanh, ReLU, and their variants
- [ ] Designing loss functions: output units from a maximum-likelihood perspective
- [ ] The backpropagation algorithm: the chain rule and computational graphs
- [ ] Symbolic derivation of forward and backward propagation
- [ ] Numerical gradient checking and efficient implementation

### Part III Regularization and Optimization

- [ ] Regularization: L2 parameter penalties and weight decay
- [ ] Regularization: L1 regularization and sparse representations
- [ ] Dataset augmentation and noise robustness
- [ ] Early stopping and multi-task learning
- [ ] Dropout and DropConnect
- [ ] Batch normalization and layer normalization
- [ ] Parameter tying and parameter sharing
- [ ] Adversarial training and tangent classifiers on manifolds
- [ ] Optimization problems: ill-conditioning, local minima, and saddle points
- [ ] Stochastic gradient descent and minibatch strategies
- [ ] Momentum and Nesterov accelerated gradient
- [ ] Learning-rate schedules: decay, warmup, and cosine annealing
- [ ] Adaptive learning rates: AdaGrad and RMSProp
- [ ] Adam and its variants (AdamW, AMSGrad)
- [ ] Second-order optimization approximations: Newton's method and quasi-Newton methods
- [ ] Parameter initialization: Xavier and Kaiming initialization
- [ ] Vanishing gradients, exploding gradients, and gradient clipping

### Part IV Convolutional Neural Networks

- [ ] The convolution operation and cross-correlation
- [ ] Three motivations for convolution: sparse interactions, parameter sharing, equivariant representations
- [ ] Padding, strides, and receptive fields
- [ ] Pooling: max pooling and average pooling
- [ ] Multiple input/output channels and 1×1 convolutions
- [ ] LeNet: the first successful convolutional network
- [ ] AlexNet: the beginning of deep convolutional networks
- [ ] VGG: block-based networks and depth exploration
- [ ] NiN: networks within networks and global average pooling
- [ ] GoogLeNet: the parallel Inception architecture
- [ ] ResNet: residual connections and identity mappings
- [ ] DenseNet: dense connections for feature reuse
- [ ] Depthwise separable convolutions and MobileNet's lightweight design
- [ ] Visualizing convolutional networks and interpreting features

### Part V Sequence Modeling

- [ ] Sequence data and language-model fundamentals
- [ ] RNN structure and forward computation
- [ ] Backpropagation through time (BPTT)
- [ ] Truncating sequences and gradient computation strategies
- [ ] LSTM: gated memory and long-range dependencies
- [ ] GRU: gated recurrent units
- [ ] Deep RNNs and bidirectional RNNs
- [ ] Sampling from sequence models and text generation
- [ ] Sequence-to-sequence learning (Seq2Seq) and encoder–decoder architectures
- [ ] Beam search
- [ ] Attention mechanisms: Bahdanau and Luong attention
- [ ] Attention pooling: a Nadaraya-Watson kernel-regression view
- [ ] Self-attention and scaled dot-product attention

### Part VI Transformers and Pretraining

- [ ] Dissecting the overall Transformer architecture
- [ ] Multi-head attention
- [ ] Positional encoding: sinusoidal and learned encodings
- [ ] Advanced positional encodings: RoPE and ALiBi
- [ ] Transformer encoders, decoders, and cross-attention
- [ ] Feedforward networks, residual connections, and normalization placement (Pre-LN/Post-LN)
- [ ] BERT: masked language modeling and pretrained representations
- [ ] The GPT family: autoregressive language modeling
- [ ] Designing pretraining tasks: MLM, NSP, and causal language modeling
- [ ] Fine-tuning on downstream tasks and prompt learning
- [ ] Scaling laws and emergent abilities of large language models

### Part VII Representation Learning and Generative Models

- [ ] Representation learning: distributed representations and embeddings
- [ ] Word embeddings: Word2Vec and GloVe
- [ ] Autoencoders: undercomplete, sparse, and denoising autoencoders
- [ ] Variational autoencoders (VAEs) and the reparameterization trick
- [ ] Generative adversarial networks (GANs): the game and training dynamics
- [ ] Improvements to GANs: WGAN, conditional GANs, and mode collapse
- [ ] Diffusion models: forward noising and reverse denoising
- [ ] DDPM and DDIM: accelerating sampling
- [ ] Flow models and normalizing flows
- [ ] Self-supervised learning: constructing pretraining tasks
- [ ] Contrastive learning: SimCLR and MoCo
- [ ] Masked autoencoders (MAEs) and masked image modeling

### Part VIII Vision and Multimodal Extensions

- [ ] Vision Transformer (ViT): patch embeddings for images
- [ ] Swin Transformer: shifted windows and a hierarchical structure
- [ ] Object detection basics: bounding boxes, anchor boxes, and non-maximum suppression
- [ ] The R-CNN family and single-stage detectors (SSD, YOLO)
- [ ] Semantic segmentation with fully convolutional networks and U-Net
- [ ] Multimodal learning: CLIP and image–text alignment
- [ ] Neural style transfer and classic image-generation applications

### Part IX Practical Methodology and Engineering

- [ ] Performance metrics and default baseline models
- [ ] Data preprocessing: normalization, standardization, and augmentation strategies
- [ ] Hyperparameter tuning: grid search, random search, and Bayesian optimization
- [ ] Debugging strategies: from single-sample overfitting to end-to-end checks
- [ ] Mixed-precision training and gradient scaling
- [ ] Distributed training: data parallelism and model parallelism
- [ ] Multi-GPU training and gradient synchronization
- [ ] PyTorch engineering: data loading with Dataset/DataLoader
- [ ] PyTorch engineering: model definition, training loops, and checkpoints
- [ ] Model deployment: ONNX export and inference optimization
- [ ] Experiment management: logging, reproducibility, and random-seed control

> After writing: create a new `xxx.md` in this directory, then change the corresponding item above to `- [x] [title](./xxx)`.
