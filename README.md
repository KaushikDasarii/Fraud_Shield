````md
# FraudShield – Real-Time Credit Card Fraud Detection System

FraudShield is an end-to-end machine learning-powered fraud detection system designed to identify fraudulent credit card transactions using highly imbalanced financial datasets.

The project combines advanced classification algorithms, explainable AI, imbalance handling, and an interactive Streamlit dashboard to simulate real-world fraud analytics and risk assessment workflows used in fintech and banking systems.

---

# Problem Statement

Credit card fraud leads to billions of dollars in financial losses every year. Traditional rule-based fraud detection systems often fail to detect sophisticated fraud patterns and generate excessive false positives.

FraudShield addresses this challenge using machine learning models trained on historical transaction data to accurately detect fraudulent transactions while improving fraud recall and interpretability.

---

# Key Features

- Real-time fraud prediction system
- Fraud detection using XGBoost and Random Forest
- Handling highly imbalanced transaction datasets
- SMOTE-based oversampling and class balancing
- SHAP-based Explainable AI
- Fraud risk analysis and interpretation
- Business cost-benefit simulation
- Interactive Streamlit analytics dashboard
- ROC-AUC and PR-AUC performance evaluation
- Transaction-level fraud probability scoring

---

# Tech Stack

## Programming Language
- Python

## Machine Learning & AI
- Scikit-learn
- XGBoost
- Random Forest
- imbalanced-learn (SMOTE)
- SHAP

## Data Processing
- Pandas
- NumPy

## Visualization
- Matplotlib
- Seaborn
- Plotly

## Deployment & Dashboard
- Streamlit

---

# Dataset

Dataset Used:
- IEEE-CIS Fraud Detection Dataset / Kaggle Credit Card Fraud Dataset

Key Characteristics:
- Highly imbalanced fraud distribution
- 284K+ financial transactions
- Fraud transactions represent less than 1% of data

---

# Machine Learning Workflow

1. Data Loading & Cleaning
2. Missing Value Handling
3. Feature Engineering
4. Label Encoding
5. Feature Scaling
6. SMOTE Oversampling
7. Model Training
8. Model Evaluation
9. SHAP Explainability
10. Dashboard Deployment

---

# Models Used

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline Model |
| Random Forest | Ensemble Comparison |
| XGBoost | Primary Fraud Detection Model |

---

# Evaluation Metrics

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

# Explainable AI

FraudShield integrates SHAP (SHapley Additive Explanations) to provide transparent fraud prediction explanations.

The system explains:
- Why a transaction was flagged
- Feature contribution importance
- Risk-driving transaction patterns

Important fraud indicators include:
- Transaction Amount
- Transaction Frequency
- HourOfDay
- Amount-to-Mean Ratio

---

# Streamlit Dashboard Features

The interactive dashboard includes:

- Real-time fraud prediction
- Fraud probability scoring
- Transaction explorer
- Risk segmentation system
- Fraud analytics visualizations
- SHAP explanation plots
- ROC & PR performance charts
- Business cost-benefit simulator

---

# Project Structure

```bash
FraudShield/
│
├── app.py
├── requirements.txt
├── main.py
│
├── data/
│   ├── transactions.csv
│   └── fraud_sample.csv
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── visualize.py
│
├── dashboard/
│   └── streamlit_app.py
│
├── models/
│   └── xgboost_model.pkl
│
└── README.md
````

---

# Installation

## Clone Repository

```bash
git clone https://github.com/KaushikDasarii/Fraud_Shield.git

cd Fraud_Shield
```

---

# Create Virtual Environment

## Windows

```bash
python -m venv venv

venv\Scripts\activate
```

## Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Model Training

```bash
python main.py
```

---

# Launch Streamlit Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

---

# Results

* Successfully detected fraudulent transactions using ensemble learning models
* Achieved strong ROC-AUC and fraud recall performance
* Built explainable fraud prediction workflows using SHAP
* Developed an interactive fraud analytics dashboard
* Simulated real-world fintech fraud monitoring systems

---

# Future Improvements

* Real-time API integration
* Kafka streaming pipeline
* Deep learning fraud detection
* Behavioral analytics
* Geolocation anomaly detection
* MLOps deployment pipeline
* MLflow experiment tracking

---

# Skills Demonstrated

* Machine Learning
* Fraud Analytics
* Explainable AI
* Ensemble Learning
* Imbalanced Classification
* Feature Engineering
* Model Evaluation
* Streamlit Deployment
* Business Analytics
* Real-Time Prediction Systems

---

# Author

## Kaushik Dasari

### LinkedIn

https://www.linkedin.com/in/dasari-hari-venkat-kaushik-5061a6342

### GitHub

https://github.com/KaushikDasarii

```
```
