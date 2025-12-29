# Fraud Risk Scoring with Neural Networks (Credit Card Transactions)

## Overview
This project implements an end-to-end **Fraud Risk Scoring System** using real-world credit card transaction data.  
Instead of producing a simple binary decision (fraud / not fraud), the system assigns a **probability score** to each transaction, reflecting its estimated fraud risk.

The project is designed and implemented with a **Machine Learning Engineering mindset**:
- data leakage prevention
- reproducible pipelines
- appropriate evaluation for highly imbalanced data
- clear separation between exploration, preprocessing, modeling, and evaluation

The dataset is based on anonymized credit card transactions and reflects realistic challenges found in financial fraud detection.

---

## Problem Context
Credit card fraud detection is a classic **high-risk, low-frequency** problem:
- Fraudulent transactions are extremely rare (< 0.2%)
- False positives directly impact customer experience
- False negatives lead to financial loss and regulatory risk

This project focuses on **risk scoring**, not naive classification accuracy, which would be misleading in such an imbalanced setting.

---

## Dataset
- Source: Kaggle – Credit Card Fraud Detection
- Transactions: 284,807
- Fraud cases: 492 (≈ 0.17%)
- Features:
  - `V1–V28`: anonymized PCA-transformed features
  - `Time`: seconds since first transaction
  - `Amount`: transaction value
  - `Class`: target label (0 = legitimate, 1 = fraud)

The data is intentionally anonymized, reflecting real-world privacy constraints.

---

## Project Structure

fraud-risk-scoring/
├── data/
│   ├── raw/                 # Original dataset (not versioned)
│   └── processed/           # Preprocessed datasets
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baselines.ipynb
│   ├── 04_nn_training.ipynb
│   ├── 05_evaluation_thresholding.ipynb
│   └── 06_explainability.ipynb
│
├── src/                     # Reusable pipeline components
│
├── models/                  # Trained models & scalers (ignored by Git)
│
├── reports/
│   ├── figures/             # Plots and visualizations
│   └── metrics/             # Evaluation results
│
├── docs/
│   ├── model_card.md
│   ├── risk_register.md
│   └── decision_log.md
│
└── README.md



## Workflow & Methodology

### 1. Exploratory Data Analysis (EDA)
- Verified data integrity and schema
- Quantified extreme class imbalance
- Analyzed distributions of `Amount` and `Time`
- Confirmed PCA feature properties
- Identified key modeling risks early

### 2. Preprocessing (Leakage-Safe)
- Deterministic train/test split with stratification
- Scaling applied **only** to `Time` and `Amount`
- Scaler fitted exclusively on training data
- Explicit leakage checks and sanity validations
- Reproducibility ensured via fixed random seeds

### 3. Baseline Modeling
- Logistic Regression used as a strong, interpretable baseline
- Performance evaluated with:
  - Precision
  - Recall
  - PR-AUC
- Accuracy intentionally avoided due to class imbalance

### 4. Neural Network Risk Model
- Small, fully connected neural network
- Sigmoid output for probability-based risk scoring
- Regularization via dropout and early stopping
- Architecture designed to avoid overfitting on rare fraud cases

### 5. Evaluation & Thresholding
- Precision–Recall curves analyzed
- Decision thresholds chosen based on business trade-offs
- Explicit discussion of false positives vs. false negatives
- Output designed for analyst review and operational decision-making

### 6. Explainability & Governance
- Feature contribution analysis (model-level explainability)
- Documented limitations and risks
- Clear separation between technical output and business decisions
- Model artifacts handled in a reproducible, auditable manner

---

## Key Results
- Successfully built a fraud risk scoring system producing calibrated probabilities
- Demonstrated why accuracy is misleading in fraud detection
- Showed how small, regularized models can generalize better than complex architectures
- Implemented industry-relevant evaluation and decision logic

---

## Engineering & Quality Principles
- No data leakage
- Reproducible pipelines
- Clear separation of concerns
- Audit-ready structure
- Emphasis on interpretation and decision impact, not just metrics

---

## Limitations & Next Steps
- Dataset is static and anonymized
- No real-time deployment in current version
- Potential extensions:
  - Cost-sensitive learning
  - Concept drift monitoring
  - Integration into streaming pipelines
  - Advanced explainability techniques

---

## Motivation
This project was built to bridge the gap between **academic machine learning** and **real-world risk modeling**, with a strong focus on correctness, transparency, and engineering discipline.



