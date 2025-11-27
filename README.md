Machine Learning Classification Projects
Binary and Multiclass Classification

This repository contains two machine-learning projects implemented with clean preprocessing pipelines, multiple classical and deep-learning models, and detailed comparative analysis.

Projects included:
1. Binary Classification – Startup Founder Retention Prediction
2. Multiclass Classification – Personality Cluster Prediction

Each project lives in its own folder with dedicated notebooks, scripts, and reports.

Repository Structure:
.
├── Binary_Classification/
│   ├── data/
│   ├── notebooks/
│   ├── models/
│   ├── src/
│   └── report.pdf
├── Multiclass_Classification/
│   ├── data/
│   ├── notebooks/
│   ├── models/
│   ├── src/
│   └── report.pdf
└── README.md

-------------------------
1. Binary Classification  
Startup Founder Retention Prediction

Objective:
Predict whether a founder will Stay or Leave a startup using demographic, organisational, and operational features.

Dataset:
- 59,611 training rows, 24 features
- Target: Stayed vs Left
- Mild imbalance, so Macro F1 is used throughout

Preprocessing Summary:
- Median imputation for numeric and ordinal features
- Mode imputation for nominal features
- Boolean normalization (Yes/No → 1/0)
- Manual ordinal mapping for ordered categories
- Outlier capping using IQR
- Log-transform for skewed numerics
- Standard scaling (classical models), Robust scaling (NN)

Important Features:
- startup stage
- personal status
- remote operations
- work–life balance rating
- startup reputation
- funding rounds led

Models and Performance:
Logistic Regression: 0.736
SVM (RBF Kernel): 0.751
Neural Network (MLP): 0.747
CatBoost: 0.766

-------------------------
2. Multiclass Classification  
Personality Cluster Prediction

Objective:
Predict personality cluster (A–E) using behavioural, demographic, activity-based and environmental features.

Dataset:
- 1,913 cleaned training rows
- Classes: A, B, C, D, E
- Highly imbalanced

Preprocessing Summary:
- Median imputation for numeric/ordinal
- Mode imputation for nominal
- One-hot encoding for nominal
- Ordinal features preserved as integers
- Standard scaling

Feature Insights:
- consistency score (strongest predictor)
- focus intensity
- hobby engagement level
- creative expression index

Models and Performance:
Logistic Regression: 0.478
SVM (RBF Kernel): 0.606
Neural Network (MLP): 0.636
CatBoost: 0.589

-------------------------
How to Run:

Install Dependencies:
pip install -r requirements.txt

Run Jupyter Notebooks:
Each project contains notebooks for EDA, preprocessing, training and evaluation.

Run Training Scripts:
python Binary_Classification/src/model.py

-------------------------
Reports:
ML_Binary.pdf
ML_Multi.pdf

-------------------------
Summary:
- Strong classical and deep-learning baselines for binary classification
- Neural network performs best on the multiclass task
- Consistent preprocessing and evaluation framework across both projects
