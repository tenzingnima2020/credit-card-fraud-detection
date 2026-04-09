# 💳 Credit Card Fraud Detection System

🚀 A Machine Learning project to detect fraudulent credit card transactions with an interactive Streamlit web app.

---

## 📌 Project Overview

This project builds a **fraud detection system** using machine learning techniques.  
It analyzes transaction data, applies feature engineering, and trains models to identify fraudulent activity with high accuracy.

---

## 🎯 Objectives

- Detect fraudulent transactions using ML models  
- Handle highly imbalanced datasets  
- Improve model performance using feature engineering  
- Evaluate models using Precision, Recall, and F1-score  
- Deploy an interactive **Streamlit web application**

---

## 📊 Dataset

The dataset contains transaction-level data with:

- 💰 Transaction Amount  
- ⏱️ Time  
- 🔢 Engineered features (V1–V28)  
- ⚠️ Highly imbalanced classes (fraud is rare)

---

## 🛠️ Technologies Used

- 🐍 Python  
- 📊 Pandas, NumPy  
- 🤖 Scikit-learn  
- ⚡ XGBoost  
- 📈 Matplotlib / Plotly  
- 🌐 Streamlit  

---

## ⚙️ Project Workflow

### 1. Data Preprocessing
- Handling missing values  
- Feature scaling  
- Train-test split  

### 2. Feature Engineering
- Created new features:
  - `is_night` (fraud more likely at night)  
  - Transaction patterns  

### 3. Model Building
- Logistic Regression  
- Random Forest  
- XGBoost (**Best performing model**)  

### 4. Model Evaluation
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  

---

## 📈 Key Results

- ✅ Improved fraud detection using engineered features  
- 🎯 High Recall achieved (important for fraud detection)  
- ⚖️ Balanced precision and recall  

---

## 🖥️ Streamlit Application

The app allows users to:

- Enter transaction details  
- Predict fraud in real-time  
- Visualize model performance  

▶️ Run the app locally:

```bash
streamlit run app.py