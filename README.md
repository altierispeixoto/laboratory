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

This repository is a collection of data science projects and experiments covering various domains including:

- 📈 Stock Market Analysis
- 🏠 Housing Price Prediction
- 🚗 Car Crash Analysis
- 👥 Customer Segmentation
- 🔄 Churn Prediction
- 🤖 Machine Learning Algorithms Implementation

Each project is self-contained in its own directory with detailed documentation and analysis.

## 🗂️ Project Structure

```
laboratory/
├── boston-houses/          # Housing price prediction using linear regression
├── brazilian-stock-analysis/ # Brazilian stock market analysis and portfolio optimization
├── car-crash-areas/       # Analysis of car crash patterns and clustering
├── churn/                 # Customer churn prediction models
├── classification/        # Implementation of various classification algorithms
├── clustering/           # Implementation of clustering algorithms
└── [other-projects]/     # Additional projects and experiments
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

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
uv pip install -r requirements.txt
```

#### Option B: Using traditional pip
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

The `uv` method is recommended for:
- 🚀 Faster package installation (up to 10-100x)
- 📦 More reliable dependency resolution
- 🔒 Improved security with package verification
- 💨 Efficient caching of wheel files

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
