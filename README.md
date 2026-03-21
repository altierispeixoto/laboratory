# Data Science Laboratory

<p align="center">
    <img src="https://botacademy.s3.eu-central-1.amazonaws.com/9999_channel_design/logo/900x900.png" alt="Logo" width="150"/></a>
</p>

<div align="center">

[![MIT License](https://img.shields.io/badge/license-MIT-3C93B4.svg?style=flat)](http://choosealicense.com/licenses/mit/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)

</div>

## Table of Contents

- [About](#-about)
- [Project Structure](#️-project-structure)
- [Getting Started](#-getting-started)
- [Dependencies](#-dependencies)

## 📋 About

This repository is a comprehensive collection of data science and machine learning projects, organized into four main categories:

### 🏢 Business Cases
Real-world applications and industry-specific analyses, including:
- 📈 Stock Market Analysis & Portfolio Optimization
- 🚗 Car Crash Pattern Analysis
- 🔄 Customer Churn Prediction
- 👥 Customer Segmentation
- 🤖 ML Model Deployment
- 🔢 Mixed Integer Programming
- 🔍 Web Scraping & Data Collection

### 🎓 Courses
Structured learning paths and course materials:
- 🔥 PyTorch Professional: structured course covering foundations, advanced architectures, optimization, and deployment (Modules 1–4)

### 📚 Foundations
Core machine learning concepts and fundamental algorithms:
- 📊 Classification & Clustering
- 📐 Linear Algebra & Regression
- 📏 Distance Metrics & DTW
- 🔬 Feature Selection & Dimensionality Reduction
- 📝 Text Analysis (TF-IDF)
- 🎯 Supervised Learning

### 🚀 Advanced
Complex models and cutting-edge techniques:
- 🧠 Neural Networks & Deep Learning
- 🕸️ Complex Networks & Graph Analytics
- 🔄 Distributed Computing with PySpark
- 🎯 AutoML with PyCaret
- 🧬 Genetic Algorithms
- 📊 Advanced Clustering

## 🗂️ Project Structure

```
laboratory/
├── business-cases/      # Real-world applications and analyses
│   ├── brazilian-stock-analysis/    # Stock market analysis and portfolio optimization
│   ├── car-crash-areas/            # Geospatial analysis of accident patterns
│   ├── churn/                      # Customer churn prediction
│   ├── customer-segments/          # Customer segmentation analysis
│   ├── disney/                     # Disney-related data analysis
│   ├── hugby-fantasy/              # Fantasy sports analytics
│   ├── marketing-campaign/         # Marketing analytics
│   ├── mixed-integer-linear-programming/  # Optimization problems
│   ├── ml-api/                     # ML model deployment API
│   ├── package-inserts/            # Medical package insert analysis
│   ├── pokemon/                    # Pokémon data analysis
│   ├── smartcab-reinforcement-learning/   # RL for autonomous driving
│   └── trademe/                    # Trade analysis
│
├── courses/             # Structured course materials
│   └── pytorch-professional/       # PyTorch course (Modules 1–4)
│
├── foundations/         # Core ML concepts and algorithms
│   ├── classification/             # Basic classification algorithms
│   ├── clustering/                 # Clustering implementations
│   ├── data-normalization/         # Data preprocessing
│   ├── distances/                  # Distance metrics
│   ├── dynamic-time-warping/       # Time series analysis
│   ├── feature-selection/          # Feature engineering
│   ├── iris/                       # Classic ML dataset
│   ├── linear-algebra/             # Mathematical foundations
│   ├── linear-regression/          # Regression techniques
│   ├── reduction-dimensionality/   # Dimension reduction
│   ├── regression/                 # Advanced regression
│   ├── student-intervention/       # Educational data mining
│   └── tf-idf/                     # Text feature engineering
│
├── advanced/            # Advanced techniques and implementations
│   ├── complex-networks/           # Graph analytics
│   ├── embeddings/                 # Vector representations
│   ├── experimental/               # Research implementations
│   ├── exponential-backoff/        # Retry mechanisms
│   ├── fuzzy-clustering/           # Fuzzy logic clustering
│   ├── genetic-algorithm-feat-selection/  # Feature selection
│   ├── gradient-descent/           # Optimization algorithms
│   ├── human-activity-recognition/ # Deep learning
│   ├── neural-networks-from-scratch/  # NN implementations
│   ├── pycaret/                    # AutoML experiments
│   └── pyspark+sklearn/            # Distributed ML
│
└── miscellaneous/       # Configuration, setup, and reference materials
    ├── config/                     # Environment configuration
    ├── my-setup/                   # Personal tooling setup
    ├── papers/                     # Research papers
    └── proxy-rotating/             # Proxy utilities
```

## 🚀 Getting Started

1. Clone the repository
```bash
git clone https://github.com/altierispeixoto/laboratory.git
cd laboratory
```

2. Set up the environment using [uv](https://github.com/astral-sh/uv) (Recommended)

[uv](https://github.com/astral-sh/uv) is a modern Python package installer and resolver written in Rust, offering significantly faster installation speeds.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
uv sync
```

## 📦 Dependencies

Each project may have its own specific requirements, but the general dependencies are:

- Python 3.12+
- Jupyter Notebook/Lab
- Common Data Science Libraries:
  - NumPy
  - Pandas
  - Scikit-learn
  - Matplotlib
  - Seaborn
  - TensorFlow/PyTorch (for deep learning projects)

---

⭐ If you find this repository useful, please consider giving it a star!
