# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This project uses `uv` for Python package management (Python 3.12+):

```bash
uv venv && source .venv/bin/activate && uv sync
```

For Jupyter work:
```bash
jupyter notebook  # or jupyter lab
```

## Repository Structure

This is an educational ML/data science lab with four main areas:

- **`foundations/`** — Core ML algorithms (classification, clustering, regression, dimensionality reduction, etc.)
- **`advanced/`** — Complex techniques (neural networks from scratch, graph NNs, genetic algorithms, PySpark, AutoML)
- **`business-cases/`** — Real-world applications (stock analysis, churn prediction, NLP, RL, etc.)
- **`courses/pytorch-professional/`** — Structured PyTorch course across 4 modules:
  - Module 1: Foundations
  - Module 2: Advanced Architectures (custom, vision, NLP)
  - Module 3: Optimization
  - Module 4: Preparing Models for Deployment (MLflow, ONNX, quantization, pruning)

## Tech Stack

- **Deep Learning:** PyTorch + PyTorch Lightning, TensorFlow
- **ML:** Scikit-learn, PyCaret, fastai
- **NLP/Vision:** HuggingFace Transformers, diffusers, torchvision, OpenCV
- **Graph ML:** torch-geometric
- **MLOps:** MLflow, ONNX + onnxruntime, onnx-tf
- **Data:** Pandas, NumPy, PyArrow
- **Distributed:** PySpark
- **Optimization:** Optuna

## Working with Notebooks

Most work happens in `.ipynb` files. Use `NotebookEdit` tool to modify cells. Each subdirectory typically contains one or more notebooks alongside data files or supporting Python modules.

The `courses/pytorch-professional/` assignments follow a consistent pattern: lecture notebooks for reference and an `assignment/` subdirectory with the actual work (`C{course}M{module}_Assignment.ipynb`).
