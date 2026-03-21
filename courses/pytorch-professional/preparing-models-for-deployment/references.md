## Module 4 Resources

### Model serialization and versioning
- PyTorch Checkpointing Tutorial. Official guide to saving and loading models reliably. [Link](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- Weights & Biases: Model Versioning Guide. Practical strategies for naming, metadata, and reproducibility. [Link](https://wandb.ai/site/articles/intro-to-mlops-data-and-model-versioning/)
- Determinism in Deep Learning (PyTorch Blog). Discusses reproducibility challenges and best practices for experiments. [Link](https://docs.pytorch.org/docs/stable/notes/randomness.html)

## MLflow
- MLflow: Open Source Platform for ML Lifecycle. Official overview of tracking, projects, and model registry. [Link](https://mlflow.org/)
- MLflow with PyTorch. Walkthrough on logging PyTorch models with MLflow. [Link](https://mlflow.org/docs/3.1.3/ml/deep-learning/pytorch/quickstart/quickstart-pytorch/)

## ONNX and portability
- ONNX Runtime Documentation. Explains running ONNX models across hardware and frameworks. [Link](https://onnxruntime.ai/docs/tutorials/accelerate-pytorch/pytorch.html)
- Exporting PyTorch Models to ONNX (PyTorch Docs). Step-by-step process for exporting and validating models. [Link](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)
  
## Pruning
- Neural Network Pruning Explained (Paperspace Blog) — Clear taxonomy, motivations, and step-by-step pruning workflows. [Link](https://blog.paperspace.com/neural-network-pruning-explained/)
- A Comprehensive Guide to Neural Network Model Pruning (Datature) — Compares structured vs. unstructured pruning, trade-offs, and deployment considerations. [Link](https://datature.io/blog/a-comprehensive-guide-to-neural-network-model-pruning)

## Static & dynamic quantization
- Hugging Face Blog: bitsandbytes Integration (8-bit Inference) — Practical low-precision inference overview, memory savings, and speedups. [Link](https://huggingface.co/blog/hf-bitsandbytes-integration)
- A Visual Guide to Quantization (Maarten Grootendorst) — Intuitive diagrams explaining scales, zero-points, and precision trade-offs. [Link](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)

## Quantization-Aware Training (QAT)
- Quantization-Aware Training: Efficient AI on Edge Devices (W&B Report) — End-to-end QAT pipeline, experiments, and practical recommendations. [Link](https://wandb.ai/onlineinference/qat/reports/Quantization-Aware-Training-Empowering-efficient-AI-on-edge-devices--VmlldzoxMTcyOTEwMA)
- QAT: Step-by-Step Guide with PyTorch (W&B Report) — Detailed walkthrough of preparing, training, and converting QAT models. [Link](https://wandb.ai/byyoung3/Generative-AI/reports/Quantization-Aware-Training-QAT-A-step-by-step-guide-with-PyTorch--VmlldzoxMTk2NTY2Mw)
- Quantization and Training for Integer-Only Inference (Jacob et al.) — Mathematical foundations for fake-quantization, calibration, and integer inference. [Link](https://arxiv.org/abs/1712.05877)
