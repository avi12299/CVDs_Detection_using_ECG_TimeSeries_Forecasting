# 🫀 CVDs Detection using ECG Time-Series Forecasting and Deep Learning Models

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/PyTorch_Lightning-1.5+-purple.svg)](https://www.pytorchlightning.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Project Overview

This project implements a comprehensive framework for **Cardiovascular Disease (CVD) detection** using **ECG time-series forecasting** with state-of-the-art deep learning models and transformer architectures. The system leverages advanced neural networks to analyze ECG signals and predict cardiovascular conditions through time-series pattern recognition.

### 🌟 Key Features

- **Multi-Model Architecture**: Implementation of 8+ deep learning models including Chronos, Moirai, TimeGPT, LSTM, GRU, BiLSTM, RNN, and CNN
- **Advanced Time-Series Forecasting**: Specialized for ECG signal prediction and anomaly detection
- **Comprehensive Evaluation**: Multiple metrics (MSE, RMSE, MAE, MAPE, R²) with detailed visualizations
- **Automated Training Pipeline**: PyTorch Lightning-based training with early stopping and learning rate scheduling
- **Rich Visualization Suite**: Scatter plots, line plots, box plots, and comprehensive result analysis
- **Scalable Dataset Processing**: Support for multiple patient datasets (P1-P5)

## 🏗️ Project Structure

```
CVDs_Detection_using_ECG_TimeSeries_Forecasting/
├── 📁 Datasets/                     # ECG Time-Series Data
│   ├── P1.xlsx                      # Patient 1 ECG data
│   ├── P2.xlsx                      # Patient 2 ECG data
│   ├── P3.xlsx                      # Patient 3 ECG data
│   ├── P4.xlsx                      # Patient 4 ECG data
│   └── P5.xlsx                      # Patient 5 ECG data
├── 📄 P_1_main.ipynb               # Main implementation notebook
├── 📁 results/                      # Model predictions and metrics
├── 📁 boxplots/                     # Box plot visualizations
├── 📁 scatter_plots/                # Scatter plot visualizations
├── 📁 line_plots_1/                 # Line plot visualizations
├── 📁 line_plots_orange/            # Alternative line plots
├── 📄 results.csv                   # Comprehensive results summary
├── 📄 skip_list.json               # Training optimization cache
└── 📋 README.md                     # Project documentation
```

## 🧠 Model Architecture

### Deep Learning Models Implemented

| Model | Architecture | Key Features |
|-------|-------------|-------------|
| **Chronos** | LSTM-based Transformer| Advanced dropout, multi-layer LSTM |
| **Moirai** | GRU-based Transformer| Efficient gated recurrent units |
| **TimeGPT** | LSTM Transformer | GPT-inspired time-series modeling |
| **LSTM** | Long Short-Term Memory | Classic sequential modeling |
| **GRU** | Gated Recurrent Unit | Simplified gating mechanism |
| **BiLSTM** | Bidirectional LSTM | Forward-backward processing |
| **RNN** | Vanilla RNN | Basic recurrent architecture |
| **CNN** | 1D Convolutional | Feature extraction via convolution |

### Model Configuration
- **Input Size**: 12 time steps
- **Output Size**: 1 prediction
- **Hidden Units**: 64
- **Layers**: 3
- **Batch Size**: 32
- **Max Epochs**: 50
- **Early Stopping**: 15 patience epochs

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.8+
CUDA-compatible GPU (recommended)
```

### Install Dependencies
```bash
pip install torch torchvision torchaudio
pip install pytorch-lightning
pip install pandas numpy scikit-learn
pip install matplotlib seaborn
pip install xlrd openpyxl
```

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/CVDs_Detection_using_ECG_TimeSeries_Forecasting.git
cd CVDs_Detection_using_ECG_TimeSeries_Forecasting

# Install requirements
pip install -r requirements.txt

# Run the main script
python P_1_main.py
```

## 📊 Dataset Information

The project uses ECG time-series data from 5 patients (P1-P5), each containing:
- **Datetime**: Timestamp information
- **ii**: ECG signal values (Lead II)
- **Preprocessing**: MinMax normalization, forward-fill for missing values
- **Split**: 60% training, 20% validation, 20% testing

### Data Processing Pipeline
```python
# Data preprocessing workflow
1. Load Excel files with datetime and ECG values
2. Handle missing values using forward-fill
3. Apply MinMax scaling (0-1 normalization)
4. Create sliding window sequences (input_size=12)
5. Split into train/validation/test sets
```


## 📈 Performance Metrics

The framework evaluates models using comprehensive metrics:

### Evaluation Metrics
- **MSE (Mean Squared Error)**: L2 loss measure
- **RMSE (Root Mean Squared Error)**: Standard deviation of residuals
- **MAE (Mean Absolute Error)**: Average absolute differences
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based accuracy
- **R² (Coefficient of Determination)**: Variance explanation measure

### Visualization Suite
- **📊 Box Plots**: Statistical distribution analysis
- **📈 Scatter Plots**: Actual vs Predicted correlations
- **📉 Line Plots**: Time-series prediction visualization
- **📋 Results Tables**: Comprehensive metric summaries

## 🔬 Experimental Results

### Model Performance Comparison
Results are automatically saved and visualized for easy comparison:

```python
# Results are saved in multiple formats:
results.csv                    # Comprehensive metrics table
results/model_predictions.csv  # Individual model predictions
boxplots/                     # Statistical visualizations
scatter_plots/                # Correlation analysis
line_plots_*/                 # Time-series visualizations
```

### Key Insights
- Models are evaluated across multiple iterations for statistical significance
- Early stopping prevents overfitting
- Learning rate scheduling optimizes convergence
- Comprehensive visualization enables deep analysis

## ⚙️ Configuration Options

### Training Parameters
```python
# Configurable parameters
INPUT_SIZE = 12        # Sequence length
OUTPUT_SIZE = 1        # Prediction horizon
NUM_ITERATIONS = 8     # Statistical runs
BATCH_SIZE = 32        # Training batch size
MAX_EPOCHS = 50        # Training epochs
PATIENCE = 15          # Early stopping patience
```


## 🛠️ Advanced Features

### Smart Training Management
- **Skip List**: Avoids retraining completed models
- **Checkpointing**: Automatic model state saving
- **Memory Management**: GPU memory optimization
- **Progress Tracking**: Real-time training monitoring


## 📊 Visualization Gallery

### Generated Visualizations
1. **Box Plots**: `boxplots/all_boxplots_metrics.png`
2. **Scatter Plots**: `scatter_plots/all_last_iterations_scatter_plots.png`
3. **Line Plots**: `line_plots_*/all_last_iterations_line_plots.png`
4. **Individual Results**: Model-specific prediction visualizations


## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Development installation
git clone https://github.com/avi12299/CVDs_Detection_using_ECG_TimeSeries_Forecasting.git
cd CVDs_Detection_using_ECG_TimeSeries_Forecasting
pip install -e .
```

## 📋 Requirements

### Core Dependencies
```
torch>=1.9.0
pytorch-lightning>=1.5.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
xlrd>=2.0.0
openpyxl>=3.0.0
```

### Hardware Requirements
- **RAM**: 8GB+ recommended
- **GPU**: CUDA-compatible (optional but recommended)
- **Storage**: 2GB+ for datasets and results





## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Avinash kumar** - *Initial work* - [@avi12299](https://github.com/yourusername)

## 🙏 Acknowledgments

- PyTorch Lightning team for the excellent framework
- Contributors to the ECG datasets
- Open-source community for tool development

## 📞 Support

For support and questions:
- 📧 **Email**: avinashkr1302@gmail.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/avi12299/CVDs_Detection_using_ECG_TimeSeries_Forecasting/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/avi12299/CVDs_Detection_using_ECG_TimeSeries_Forecasting/discussions)

---

<div align="center">

**⭐ Star this repository if it helped you! ⭐**

Made with ❤️ for advancing cardiovascular health through AI

</div>
