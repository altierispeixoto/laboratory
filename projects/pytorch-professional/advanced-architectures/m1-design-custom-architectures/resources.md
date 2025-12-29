Module 1 Resources

### Modular programming & custom flows
- The Annotated Transformer (Harvard NLP). A step-by-step, modular build that showcases shared parameters, residual paths, and clean composition patterns you can reuse in PyTorch. [Link](https://nlp.seas.harvard.edu/annotated-transformer)
- Design Patterns for ML Code (Eugene Yan). Practical patterns—modularity, separation of concerns, config-driven components—that make custom architectures easier to extend and test. [Link](https://eugeneyan.com/writing/design-patterns)

### Siamese networks & other “non-standard” architectures

- Siamese Neural Networks for One-Shot Learning (Koch et al.). Introduces the pairwise comparison paradigm, dataset pairing strategy, and a simple contrastive training recipe. [Link](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
 
- Intro to Autoencoders (Jeremy Jordan). A gentle primer on undercomplete, denoising, and variational autoencoders—what they are and when to use them. [Link](https://www.jeremyjordan.me/autoencoders/)

- Sparsely-Gated Mixture-of-Experts (Shazeer et al.). A conditional-execution architecture that routes each input to a small subset of “expert” subnetworks, achieving huge capacity with near-constant compute—great inspiration for dynamic, modular designs. [Link](https://arxiv.org/abs/1701.06538)

### Contrastive / triplet objectives

- Dimensionality Reduction by Learning an Invariant Mapping (Hadsell et al.). The classic paper defining contrastive loss and how it shapes embedding spaces. [Link](https://www.researchgate.net/profile/Yann_Lecun/publication/4246277_Dimensionality_Reduction_by_Learning_an_Invariant_Mapping/links/00b7d514af9f25ecca000000/Dimensionality-Reduction-by-Learning-an-Invariant-Mapping.pdf)

- FaceNet: A Unified Embedding for Face Recognition and Clustering (Schroff et al.). Popularizes triplet loss with hard/semi-hard mining for strong identity separation. [Link](https://arxiv.org/abs/1503.03832)

- Contrastive Representation Learning (Lilian Weng). A readable survey of modern contrastive objectives, positives/negatives, and temperature tricks. [Link](https://lilianweng.github.io/posts/2021-05-31-contrastive/)

### Skip connections (math intuition)

- Identity Mappings in Deep Residual Networks (He et al.). Formalizes why identity shortcuts preserve gradient flow and stabilize very deep nets. [Link](https://arxiv.org/abs/1603.05027)

- ResNets Behave Like Ensembles of Relatively Shallow Networks (Veit et al.). Offers an ensemble-like perspective that explains robustness and trainability. [Link](https://arxiv.org/abs/1605.06431)
  
- An Intuitive Explanation of Skip Connections (The AI Summer). Derives the backward pass through a residual block and ties it to vanishing/exploding-gradient mitigation.[Link](https://theaisummer.com/skip-connections)

### DenseNet & transition layers
- Densely Connected Convolutional Networks (Huang et al.). Shows how dense connectivity improves feature reuse and gradient flow while reducing parameters. [Link](https://arxiv.org/abs/1608.06993)
- DenseNet Architecture Explained with PyTorch (Aman Arora). Code-level walkthrough of DenseBlocks and Transition Layers with growth-rate/compression trade-offs. [Link](https://amaarora.github.io/posts/2020-08-02-densenets.html)