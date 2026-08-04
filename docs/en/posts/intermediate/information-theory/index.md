---
pageClass: plain-doc
---

# Information Theory

Following the core chapters of Cover & Thomas's *Elements of Information Theory*, this study plan starts from entropy and mutual information, then covers source coding, channel capacity, rate-distortion theory, and Kolmogorov complexity, and finally extends to the connections between information theory and statistics and machine learning.

## Topic Planning

<ProgressGrid cat="intermediate/information-theory" />


### Part I Entropy, Relative Entropy, and Mutual Information

- [ ] Self-information and the definition of entropy
- [ ] Properties of entropy: non-negativity, symmetry, and additivity
- [ ] Joint Entropy and Conditional Entropy
- [ ] Chain Rule for Entropy
- [ ] Definition and properties of Relative Entropy (KL divergence)
- [ ] Definition of Mutual Information and its multiple equivalent expressions
- [ ] Conditional mutual information and the chain rule for mutual information
- [ ] Venn diagram relationships among entropy, conditional entropy, and mutual information
- [ ] Properties of relative entropy: non-negativity and Gibbs' inequality
- [ ] Convexity of relative entropy and its chain rule

### Part II Basic Inequalities and Data Processing

- [ ] Log Sum Inequality
- [ ] Jensen's inequality and its applications in information theory
- [ ] Data Processing Inequality
- [ ] Monotonicity of mutual information along Markov chains
- [ ] Sufficient Statistic and the data processing inequality
- [ ] Fano's inequality and its implications for error probability
- [ ] Uniform distribution and maximum entropy
- [ ] Conditioning Reduces Entropy

### Part III Asymptotic Equipartition Property (AEP)

- [ ] Review of the weak law of large numbers and convergence in probability
- [ ] Asymptotic Equipartition Property
- [ ] Definition and properties of the Typical Set
- [ ] Relationship between high-probability sets and typical sets
- [ ] Connection between AEP and data compression
- [ ] Joint typicality and jointly typical sequences
- [ ] Estimating the number of jointly typical sequences

### Part IV Source Coding and Data Compression

- [ ] The source coding problem and classification of codes (uniquely decodable, instantaneous, prefix-free)
- [ ] Kraft's Inequality
- [ ] Optimal code lengths and lower bounds on code length
- [ ] Shannon Code and its construction
- [ ] Construction and optimality proof of Huffman Coding
- [ ] Principles and implementation of Arithmetic Coding
- [ ] Introduction to Lempel-Ziv universal coding (LZ77/LZ78)
- [ ] Source coding theorem: the entropy bound on average code length

### Part V Channel Capacity

- [ ] The discrete memoryless channel (DMC) model
- [ ] Definition of channel capacity and its intuitive meaning
- [ ] Examples of the noiseless binary channel and noisy channels
- [ ] Capacity of the Binary Symmetric Channel
- [ ] Capacity of the Binary Erasure Channel
- [ ] Symmetric channels and simplified capacity computation
- [ ] Properties of channel capacity and approaches to solving for it

### Part VI Channel Coding Theorem — Foundations

- [ ] Jointly typical sequences and decoding methods
- [ ] The random coding idea and typical sequence decoding
- [ ] Channel coding theorem: proof of achievability
- [ ] Converse of the channel coding theorem (application of Fano's inequality)
- [ ] Source-channel separation theorem (Source-Channel Separation)
- [ ] Feedback channels and the conclusion that feedback does not increase capacity

### Part VII Differential Entropy

- [ ] Definition of Differential Entropy
- [ ] Differential entropy of uniform, exponential, and Gaussian distributions
- [ ] Difference from discrete entropy: why it can be negative
- [ ] Joint differential entropy and conditional differential entropy
- [ ] Continuous forms of relative entropy and mutual information
- [ ] Properties of differential entropy: how it transforms under change of variables
- [ ] Maximum differential entropy theorem: the Gaussian distribution maximizes entropy

### Part VIII Gaussian Channel

- [ ] Gaussian channel model and power constraint
- [ ] Derivation of Gaussian channel capacity
- [ ] Parallel Gaussian channels and water-filling (Water-filling)
- [ ] Colored noise channels and the water-filling formula
- [ ] Band-limited Gaussian channel and Shannon's Capacity Formula

### Part IX Rate-Distortion Theory

- [ ] Distortion measures and the quantization problem
- [ ] Definition of the rate-distortion function R(D)
- [ ] Properties of the rate-distortion function: monotonicity, convexity, and continuity
- [ ] Rate-distortion function for a binary source
- [ ] Rate-distortion function for a Gaussian source
- [ ] Reverse water-filling for a Gaussian source
- [ ] Rate-distortion theorem: achievability and converse
- [ ] Source-channel separation under constrained settings

### Part X Kolmogorov Complexity — Foundations

- [ ] Algorithmic information theory and description length
- [ ] Definition of Kolmogorov Complexity
- [ ] Properties of Kolmogorov complexity and its uncomputability
- [ ] Connection between Kolmogorov complexity and entropy
- [ ] Universal probability and the Solomonoff prior
- [ ] Minimum Description Length (MDL) principle

### Part XI Information Theory, Statistics, and Machine Learning

- [ ] Maximum Entropy Principle
- [ ] Deriving the maximum entropy model: distributions under moment constraints
- [ ] Relationship between the maximum entropy model and logistic regression
- [ ] Definition and properties of Fisher Information
- [ ] Cramér-Rao inequality and information inequalities
- [ ] Introduction to the Entropy Power Inequality
- [ ] Connection between Cross-Entropy and KL divergence
- [ ] The cross-entropy loss function: classification tasks from an information-theoretic view

> After writing is complete: create `xxx.md` in this directory, then change the corresponding item above to `- [x] [Title](./xxx)`.
