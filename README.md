# Neural-Network-Design
This repository contains Python implementations of key algorithms and simulations for Machine Learning and Time Series Analysis. It includes ARMA model simulations, neural network training algorithms, and a hybrid of Radial Basis Function (RBF) networks with backpropagation. The project is ideal for students, researchers, and enthusiasts looking to explore foundational algorithms in these domains.

## Files

### 1. ARMA_GPAC.py
This script implements the Generalized Partial Autocorrelation (GPAC) method for ARMA model identification. GPAC helps estimate the order of ARMA (AutoRegressive Moving Average) models based on autocorrelation functions.

**Key Features:**
- GPAC table computation
- Model order estimation for ARMA processes

### 2. ARMA_simulation_ACF_PACF.py
This script simulates ARMA processes and computes their Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF). These plots are critical for diagnosing time series data and identifying model parameters.

**Key Features:**
- ARMA model simulation
- ACF and PACF computation
- Visualization of ACF/PACF plots

### 3. Backpropagation.py
A Python implementation of the Backpropagation algorithm for training feedforward neural networks. It demonstrates the gradient descent process used to minimize the error in neural network outputs.

**Key Features:**
- Feedforward neural network creation
- Error backpropagation and weight update
- Customizable learning rate and activation functions

### 4. Perceptron_Learning_Rule.py
This script implements the classic Perceptron Learning Rule, a supervised learning algorithm for binary classification tasks. The perceptron adjusts its weights based on misclassifications until convergence is achieved.

**Key Features:**
- Binary classification using a single-layer perceptron
- Weight updates based on learning rule
- Step-by-step classification results

### 5. RBF_Backpropagation.py
This script demonstrates the combination of Radial Basis Function (RBF) networks with backpropagation. RBF networks are used for classification and regression tasks, and this script shows how backpropagation can be applied to fine-tune the network.

**Key Features:**
- RBF network implementation
- Backpropagation for RBF weight updates
- Customizable parameters for the RBF layer

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git

This project is licensed under the MIT License - see the LICENSE file for details.
