# 🕵️‍♂️ Fraud Detection with Logistic Regression on Imbalanced Data

## 📌 Project Overview

This notebook explores how to detect fraudulent transactions using a **Logistic Regression** model on a highly **imbalanced dataset**. Special attention is given to improving the model's ability to detect minority class (fraud) using **resampling techniques**, **evaluation metrics**, and **visual tools** like ROC curves.

---

## 📊 Dataset Overview

This dataset simulates real-world transactional behavior where fraud is rare but critical to catch. It consists of the following features:

### 🧾 Features

| Feature Name             | Description                                           |
|--------------------------|-------------------------------------------------------|
| `transaction_amount`     | The monetary amount of the transaction                |
| `transaction_risk_score` | A risk score associated with the transaction (scaled) |
| `is_fraud`               | Target variable (1 = Fraud, 0 = Not Fraud)            |

> ⚠️ The dataset is **heavily imbalanced** with a small percentage of fraudulent cases.

---

## 🧪 Workflow Steps

### 1. **Exploratory Data Analysis (EDA)**
- Distribution plots of features  
- Class imbalance visualization  
- Boxplots and histograms to explore feature behavior  
- Correlation heatmap

### 2. **Preprocessing**
- Addressed imbalance using **SMOTE** (Synthetic Minority Oversampling Technique)  
- Applied `StandardScaler` to normalize features  
- Splitting into train/test sets

### 3. **Model Building**
- Fitted a **Logistic Regression** model  
- Evaluated performance using:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-Score  
  - **Confusion Matrix** (visualized using Seaborn)

### 4. **ROC Curve & AUC Score**
- **ROC (Receiver Operating Characteristic) Curve** was plotted to visualize the trade-off between **True Positive Rate** and **False Positive Rate**  
- **AUC (Area Under the Curve)** was computed to summarize model performance  
  > AUC closer to 1 indicates a better classifier; near 0.5 indicates a poor one.

---

## 📈 Key Insights

- Logistic Regression performance significantly improves after **SMOTE resampling**.
- Traditional accuracy is misleading on imbalanced datasets — focus was shifted toward **recall** and **AUC**.
- **ROC-AUC analysis** provided clearer insights into the model's discriminatory power.

