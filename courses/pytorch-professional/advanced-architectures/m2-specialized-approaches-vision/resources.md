## Module 2 Resources

### Receptive field theory
- Understanding the Effective Receptive Field in Deep CNNs (Luo, Li, Urtasun). Shows that the effective receptive field is smaller and Gaussian-like, with implications for depth and kernel choices. [Link](https://arxiv.org/abs/1701.04128)
- Feature Visualization (Olah, Mordvintsev, Schubert). Visual/interactive tour of what conv nets attend to, connecting receptive fields to emergent features. [Link](https://distill.pub/2017/feature-visualization/)
- CS231n Convolutional Neural Networks for Visual Recognition (Stanford). Lecture notes explaining receptive fields through kernels, stride, and pooling. [Link](https://cs231n.github.io/convolutional-networks/#conv)

### Saliency maps: math & theory
- Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps (Simonyan et al.). Derives gradient-based saliency as ∂score/∂pixels and shows class-sensitivity. [Link](https://arxiv.org/abs/1312.6034)
- Attribution Baselines (Distill). Explains baseline choices and their mathematical consequences for attributions. [Link](https://distill.pub/2020/attribution-baselines)
- Week 5: Saliency Mapping (lecture video). Conceptual walkthrough of deconvnet, guided backprop, and Grad-CAM rationale. [Link](https://www.youtube.com/watch?v=pYaAMx_GfH0)
- XAI Methods — Saliency (blog). Discusses vanilla gradient saliency foundations and interpretive caveats clearly. [Link](https://erdem.pl/2022/02/xai-methods-saliency)

## Class Activation Maps & Grad-CAM
- Learning Deep Features for Discriminative Localization (Zhou et al., CAM). Introduces CAMs and their limitation to global-avg-pool architectures. [Link](https://arxiv.org/abs/1512.04150)

## Other interpretability techniques (vision)
- Distill: The Building Blocks of Interpretability. Conceptual primitives for interpreting deep visual representations. [Link](https://distill.pub/2018/building-blocks/)
- Network Dissection (project page). Quantifies unit–concept alignment for objects, parts, textures in CNNs. [Link](https://netdissect.csail.mit.edu/?utm_source=chatgpt.com)

## Diffusion: theory & principles (noise schedule, latent spaces, guidance)
- Lil’Log: What are Diffusion Models? (Lil’Log Blog). Beginner-friendly explanation of diffusion model intuition and training. [Link](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- Hugging Face Blog: Diffusion Models from Scratch. Intuitive derivation and minimal implementations explaining core ideas. [Link](https://huggingface.co/learn/diffusion-course/en/unit1/3)

## Diffusion models: code & hands-on
- Hugging Face Diffusers — Getting Started. Load state-of-the-art diffusion pipelines, tweak schedulers, and generate images in a few lines. [Link](https://huggingface.co/docs/diffusers/index)
- How Stable Diffusion Works (HF Course). Step-by-step notebooks explaining pipelines, schedulers, CFG, and negative prompts. [Link](https://huggingface.co/learn/courses/diffusion-models)
- lucidrains/denoising-diffusion-pytorch (repo). Minimal, readable PyTorch implementations of DDPM/DDIM and friends for learning by code. [Link](https://github.com/lucidrains/denoising-diffusion-pytorch)
- Diffusers Training Examples (from-scratch & fine-tuning). Practical scripts for training/finetuning UNets and text-to-image models with accelerators/loggers. [Link](https://github.com/huggingface/diffusers/tree/main/examples)
