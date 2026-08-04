---
pageClass: plain-doc
---

# AI for Science

Using deep learning to reshape the paradigm of scientific research: from protein structure to weather and climate, from molecular drugs to theorem proving. This post covers the core directions of AI4Science, mapped to the classic papers, courses, and monograph systems in each field.

## Topic Outline

<ProgressGrid cat="advanced/ai4science" />


### Part 1: Overview of AI4Science and Its Paradigm

- [ ] The rise of AI for Science: from experimentation, theory, and computation to the fourth paradigm of data-driven science
- [ ] Problem taxonomy and research landscape of Scientific Machine Learning (SciML)
- [ ] Characteristics of scientific data: multiscale, sparse, noisy, and physically constrained
- [ ] Symmetry and equivariance: neural network design principles from a group theory perspective
- [ ] Differentiable programming and scientific computing: the role of automatic differentiation in physical simulation
- [ ] The AI scientist workflow: hypothesis generation, experiment design, data analysis, and the automated closed loop

### Part 2: Protein Structure Prediction

- [ ] The protein folding problem: from the Anfinsen principle to the Levinthal paradox
- [ ] Multiple sequence alignment (MSA) and the extraction of co-evolutionary information
- [ ] AlphaFold2 architecture analysis: Evoformer and triangular attention updates
- [ ] The structure module of AlphaFold2: invariant point attention (IPA) and end-to-end differentiable optimization
- [ ] Confidence assessment: the meaning and interpretation of pLDDT and PAE
- [ ] ESMFold: bypassing MSA with language models for fast structure prediction
- [ ] AlphaFold3: using diffusion models to unify prediction of protein, nucleic acid, and ligand complexes
- [ ] Protein design: ProteinMPNN, RFdiffusion, and inverse folding

### Part 3: Molecules and Drug Discovery

- [ ] Molecular representations: SMILES, molecular graphs, and 3D conformational geometry
- [ ] Graph neural networks for molecular property prediction: message passing mechanisms and pretraining strategies
- [ ] Molecular fingerprints and descriptors: from Morgan fingerprints to neural fingerprints
- [ ] Molecular generation models: VAE, GAN, and diffusion-based molecular generation
- [ ] 3D molecular generation: equivariant diffusion models (EDM) and conformation generation
- [ ] Virtual screening: molecular docking, binding affinity prediction, and scoring functions
- [ ] ADMET property prediction and drug-likeness assessment
- [ ] Lead optimization: reinforcement learning-based molecular design
- [ ] Target identification and drug repositioning

### Part 4: AI and Mathematics

- [ ] Introduction to formal mathematics and the Lean theorem prover
- [ ] Automated theorem proving: from heuristic search to language model-guided proof generation
- [ ] LLM theorem proving in practice: GPT-f, LeanDojo, and DeepSeek-Prover
- [ ] FunSearch: using large models to search for new discoveries in function space
- [ ] AI-assisted mathematical conjecture: Pattern Boost and the mechanization of intuition
- [ ] Neuro-symbolic reasoning: the fusion of symbolic computation and deep learning
- [ ] AlphaGeometry and geometric theorem proving

### Part 5: AI and Physical Simulation

- [ ] Physics-informed neural networks (PINN): writing governing equations into the loss function
- [ ] Training difficulties of PINN: spectral bias, loss weighting, and causal training
- [ ] Neural operators: learning mappings between function spaces
- [ ] Detailed architecture of the Fourier neural operator (FNO)
- [ ] DeepONet and the universal operator approximation theorem
- [ ] Grid independence and multi-resolution learning
- [ ] Accelerating fluid simulation with neural networks: from Navier-Stokes to turbulence modeling
- [ ] Data-driven reduced-order models and surrogate models

### Part 6: Weather and Climate Prediction

- [ ] Traditional technical route of numerical weather prediction (NWP) and its bottlenecks
- [ ] FourCastNet: global weather forecasting based on the Fourier neural operator
- [ ] GraphCast: graph neural networks and medium-range forecasting on multiscale grids
- [ ] Pangu-Weather: 3D Earth-Specific Transformer (3DEST)
- [ ] Ensemble forecasting and probabilistic weather forecasting
- [ ] Nowcasting: short-term precipitation forecasting and radar echo extrapolation (DGMR)
- [ ] Climate modeling and AI downscaling
- [ ] Combining data assimilation with AI

### Part 7: Materials Discovery

- [ ] Materials informatics: from the Materials Genome Initiative to data-driven discovery
- [ ] Mathematical representation of crystal structures and periodicity
- [ ] Crystal graph neural networks: CGCNN and equivariant graph networks (M3GNet, CHGNet)
- [ ] GNoME: large-scale graph network active learning to discover stable crystals
- [ ] Prediction of formation energy, band gap, and elastic properties
- [ ] Interatomic potentials: machine learning force fields (MACE, NequIP)
- [ ] Inverse materials design and generative models (MatterGen, CDVAE)

### Part 8: Computational Biology

- [ ] Single-cell RNA sequencing analysis pipeline: dimensionality reduction, clustering, and cell annotation
- [ ] Single-cell foundation models: scGPT, Geneformer, and scFoundation
- [ ] Genomics language models: DNABERT, Nucleotide Transformer
- [ ] Gene regulatory network inference and chromatin accessibility prediction (Enformer)
- [ ] Variant effect prediction and pathogenicity assessment (AlphaMissense)
- [ ] Spatial transcriptomics and multimodal integration
- [ ] AI-driven drug-gene associations and precision medicine

### Part 9: AI and Chemistry

- [ ] Representation of chemical reactions: reaction SMILES and reaction graphs
- [ ] Reaction product prediction: template-based and template-free sequence-to-sequence methods
- [ ] Retrosynthetic analysis: single-step retrosynthesis models and multi-step route planning
- [ ] Reaction condition recommendation and yield prediction
- [ ] Quantum chemistry property computation and machine learning acceleration (DFT surrogate models)
- [ ] Automated chemistry laboratories: robot chemists and closed-loop experimentation

### Part 10: Scientific Foundation Models and Scientific Agents

- [ ] Overview of scientific foundation models: design principles of cross-disciplinary pretrained models
- [ ] Alignment of multimodal scientific data: text, structures, sequences, and signals
- [ ] Scientific knowledge-augmented large language models: literature mining and knowledge graphs
- [ ] Scientific agents: autonomous planning, tool invocation, and experiment execution
- [ ] Evaluation benchmarks for AI research assistants: ScienceQA, DiscoveryBench, and others
- [ ] The human-AI collaborative loop of scientific discovery

### Part 11: Differential Equations and Scientific Machine Learning

- [ ] Neural network solvers for ordinary differential equations and neural ordinary differential equations (Neural ODE)
- [ ] Comparing PDE solution paradigms: PINN, neural operators, and traditional numerical methods
- [ ] Forward and inverse problems: parameter identification and source inversion
- [ ] Learning-based solvers for multiphysics coupling problems
- [ ] Conservation laws and Hamiltonian/Lagrangian neural networks
- [ ] Reliability beyond universal approximation capacity: error estimation and uncertainty quantification
- [ ] Benchmarks and open problems in scientific machine learning

> After writing: create `xxx.md` in this directory, then change the corresponding item above to `- [x] [Title](./xxx)`.
