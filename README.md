# Data Science Laboratory

<p align="center">
    <img src="https://botacademy.s3.eu-central-1.amazonaws.com/9999_channel_design/logo/900x900.png" alt="Logo" width="150"/></a>
</p>

<div align="center">

[![MIT License](https://img.shields.io/badge/license-MIT-3C93B4.svg?style=flat)](http://choosealicense.com/licenses/mit/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)

</div>

## 📋 About

This repository is a comprehensive collection of data science and machine learning projects, organized into three main categories:

### 🏢 Business Cases
Real-world applications and industry-specific analyses, including:
- 📈 Stock Market Analysis & Portfolio Optimization
- 🚗 Car Crash Pattern Analysis
- 🔄 Customer Churn Prediction
- 👥 Customer Segmentation
- 🤖 ML Model Deployment
- � Mixed Integer Programming
- 🔍 Web Scraping & Data Collection

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

The repository is organized into three main categories, each focusing on different aspects of data science and machine learning:

```
laboratory/
├── business-cases/      # Real-world applications and analyses
│   ├── brazilian-stock-analysis/    # Stock market analysis and portfolio optimization
│   ├── car-crash-areas/            # Geospatial analysis of accident patterns
│   ├── churn/                      # Customer churn prediction
│   ├── customer-segments/          # Customer segmentation analysis
│   ├── disney/                     # Disney-related data analysis
│   ├── marketing-campaign/         # Marketing analytics
│   ├── mixed-integer-linear-programming/  # Optimization problems
│   ├── ml-api/                     # ML model deployment API
│   ├── package-inserts/            # Medical package insert analysis
│   ├── smartcab-reinforcement-learning/   # RL for autonomous driving
│   └── trademe/                    # Trade analysis
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
├── advanced/           # Advanced techniques and implementations
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
```

```

## 📦 Project Dependencies

Each project may have its own specific requirements, but the general dependencies are:

- Python 3.8+
- Jupyter Notebook/Lab
- Common Data Science Libraries:
  - NumPy
  - Pandas
  - Scikit-learn
  - Matplotlib
  - Seaborn
  - TensorFlow/PyTorch (for deep learning projects)

## 🚀 Getting Started

1. Clone the repository
```bash
git clone https://github.com/altierispeixoto/laboratory.git
cd laboratory
```

2. Set up the environment using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a modern Python package installer and resolver:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook/Lab
- Required Python packages (listed in each project's requirements)

### Installation

1. Clone the repository
```bash
git clone https://github.com/altierispeixoto/laboratory.git
cd laboratory
```

2. Set up the environment (choose one method):

#### Option A: Using uv (Recommended)
[uv](https://github.com/astral-sh/uv) is a modern Python package installer and resolver written in Rust, offering significantly faster installation speeds.

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
uv pip install -r requirements.txt
```

```

## 📊 Projects Overview

Each project directory contains:
- Jupyter notebooks with analysis
- Data files or instructions to obtain them
- Documentation of methodology and findings
- Requirements specific to the project

## 🛠️ Tools & Technologies

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, TensorFlow, PyTorch
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Deep Learning**: Neural Networks implementations
- **Statistical Analysis**: SciPy, StatsModels

```
## 📝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Altieris Peixoto - [GitHub](https://github.com/altierispeixoto)

## 🙏 Acknowledgments

- All contributors and maintainers
- Open source community for tools and libraries
- Data providers and sources used in projects

---

⭐️ If you find this repository useful, please consider giving it a star!
