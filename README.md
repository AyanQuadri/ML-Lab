# Machine Learning Classification and Clustering Project

A comprehensive collection of machine learning algorithms implemented using scikit-learn, covering both supervised and unsupervised learning techniques.

## Table of Contents

- [Machine Learning Classification and Clustering Project](#machine-learning-classification-and-clustering-project)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Algorithms Implemented](#algorithms-implemented)
    - [Supervised Learning](#supervised-learning)
      - [Classification Algorithms](#classification-algorithms)
        - [1. SVM vs Random Forest Comparison](#1-svm-vs-random-forest-comparison)
        - [2. Logistic Regression](#2-logistic-regression)
        - [3. Classification Metrics Analysis](#3-classification-metrics-analysis)
      - [Regression Algorithms](#regression-algorithms)
        - [1. Linear Regression for Classification](#1-linear-regression-for-classification)
    - [Unsupervised Learning](#unsupervised-learning)
      - [Clustering Algorithms](#clustering-algorithms)
        - [1. K-Means Clustering](#1-k-means-clustering)
        - [2. Agglomerative Clustering](#2-agglomerative-clustering)
  - [Usage](#usage)
    - [Run Individual Algorithms](#run-individual-algorithms)
  - [Expected Outputs](#expected-outputs)
    - [Classification Algorithms](#classification-algorithms-1)
    - [Regression Algorithms](#regression-algorithms-1)
    - [Clustering Algorithms](#clustering-algorithms-1)

## Overview

This Repo demonstrates various machine learning algorithms for classification, regression, and clustering tasks. Each algorithm is implemented with detailed examples, proper error handling, and comprehensive output analysis.

## Project Structure

```bash
ML-Lab/
├── supervised_learning/
│   ├── classification/
│   │   ├── svm_comparison.py          # SVM vs Random Forest comparison
│   │   ├── logistic_regression.py     # Logistic regression on Iris dataset
│   │   └── classification_metrics.py  # Comprehensive metrics evaluation
│   └── regression/
│       └── linear_regression.py       # Linear regression for classification
├── unsupervised_learning/
│   └── clustering/
│       ├── k_means.py                 # K-means clustering analysis
│       └── agglomerativ.py           # Agglomerative clustering methods
├── pyproject.toml                     # Project dependencies
├── README.md                          # This file
└── uv.lock                           # Dependency lock file
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Follow these steps to set up:

```bash
# Clone the repository
git clone https://github.com/AyanQuadri/ML-Lab.git
cd ML-Lab

# Create virtual environment
uv venv

# Activate virtual environment (optional, uv run handles this)
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
uv sync
```

## Algorithms Implemented

### Supervised Learning

#### Classification Algorithms

##### 1. [SVM vs Random Forest Comparison](supervised_learning/classification/svm_comparison.py)
- **Purpose**: Compare Support Vector Machine and Random Forest performance
- **Dataset**: Iris dataset
- **Features**: 
  - Linear SVM implementation
  - Random Forest with 100 estimators
  - Accuracy comparison
- **Run**: `uv run python supervised_learning/classification/svm_comparison.py`

##### 2. [Logistic Regression](supervised_learning/classification/logistic_regression.py)
- **Purpose**: Multi-class classification using logistic regression
- **Dataset**: Iris dataset (3 classes)
- **Features**:
  - Complete classification report
  - Dataset information display
  - Convergence optimization
- **Run**: `uv run python supervised_learning/classification/logistic_regression.py`

##### 3. [Classification Metrics Analysis](supervised_learning/classification/classification_metrics.py)
- **Purpose**: Comprehensive evaluation of classification algorithms
- **Algorithms**: Logistic Regression, Decision Tree
- **Metrics Calculated**:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - Per-class TP, FP, TN, FN values
- **Run**: `uv run python supervised_learning/classification/classification_metrics.py`

#### Regression Algorithms

##### 1. [Linear Regression for Classification](supervised_learning/regression/linear_regression.py)
- **Purpose**: Demonstrate linear regression for binary classification
- **Features**:
  - Simple binary classification example
  - Model parameter extraction
  - Threshold-based classification
- **Run**: `uv run python supervised_learning/regression/linear_regression.py`

### Unsupervised Learning

#### Clustering Algorithms

##### 1. [K-Means Clustering](unsupervised_learning/clustering/k_means.py)
- **Purpose**: Cluster analysis using K-means algorithm
- **Dataset**: Iris dataset
- **Features**:
  - 3-cluster analysis
  - Inertia calculation
  - Species distribution per cluster
  - Clustering accuracy assessment
- **Run**: `uv run python unsupervised_learning/clustering/k_means.py`

##### 2. [Agglomerative Clustering](unsupervised_learning/clustering/agglomerativ.py)
- **Purpose**: Hierarchical clustering analysis
- **Datasets**: Synthetic blob data + Iris dataset
- **Features**:
  - Multiple linkage methods (ward, complete, average, single)
  - Synthetic and real data comparison
  - Adjusted Rand Index calculation
- **Run**: `uv run python unsupervised_learning/clustering/agglomerativ.py`

## Usage

### Run Individual Algorithms

```bash
# Classification algorithms
uv run python supervised_learning/classification/svm_comparison.py
uv run python supervised_learning/classification/logistic_regression.py
uv run python supervised_learning/classification/classification_metrics.py

# Regression algorithms
uv run python supervised_learning/regression/linear_regression.py

# Clustering algorithms
uv run python unsupervised_learning/clustering/k_means.py
uv run python unsupervised_learning/clustering/agglomerativ.py
```


## Expected Outputs

### Classification Algorithms
- **SVM vs Random Forest**: Accuracy scores comparison (typically both achieve 1.0 on Iris)
- **Logistic Regression**: Detailed classification report with precision/recall for each species
- **Classification Metrics**: Comprehensive confusion matrix and per-class metrics

### Regression Algorithms
- **Linear Regression**: Predicted probability, class assignment, and model parameters

### Clustering Algorithms
- **K-Means**: Cluster centers, species distribution, clustering accuracy (~89%)
- **Agglomerative**: Multiple linkage method results, cluster purity analysis


---

**Note**: All algorithms use the Iris dataset for consistency and comparison purposes, except where synthetic data provides better demonstration of specific algorithmic properties.