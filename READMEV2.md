# 💳 Credit Card Fraud Detection System

## 📌 Short Description
This project focuses on building a machine learning model to detect fraudulent credit card transactions. The goal is to identify suspicious activities with high accuracy while minimizing false positives. The project includes data preprocessing, feature engineering, model training, and evaluation using real-world inspired financial transaction data.

---

## 🚀 Getting Started
This project demonstrates an end-to-end machine learning workflow for fraud detection. It includes:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering (including time-based features like `is_night`)
- Model training and evaluation
- Performance analysis using classification metrics

---

## 📋 Prerequisites
Make sure you have the following installed:

- Python 3.8+
- pip (Python package manager)
- Jupyter Notebook / VS Code
- Required Python libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

---

## ⚙️ Installing
1. Clone the repository:
```bash
git clone my github link
cd credit-card-fraud-detection
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Open the notebook:
```bash
jupyter notebook 
```

---

## 🧪 Running the Tests
- Run all cells in the VS Code / Jupyter Notebook sequentially.
- The notebook includes:
  - Data preprocessing steps
  - Feature creation
  - Model training
  - Model evaluation

---

## 🔍 Breakdown of Tests
The model is evaluated using classification metrics:

- **Precision** → Measures how many predicted fraud cases are actually fraud  
- **Recall** → Measures how many actual fraud cases were detected  
- **F1 Score** → Balance between precision and recall  
- **Confusion Matrix** → Shows true vs predicted classifications  

Special focus is given to **Recall**, since missing fraud cases is more critical than false alarms.

---

## 🚀 Deployment
This project can be extended for deployment using:
- Streamlit (for interactive dashboards)
- Flask / FastAPI (for API-based deployment)

To run a Streamlit app (if implemented):
```bash
streamlit run app.py
```

---

## 👤 Author
**Tenzing Nima Tamang**  
Post Graduate Student in Applied Data Analytics  
Durham College, Oshawa  

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgement
- Durham College for academic guidance  
- Open-source community for ML tools and libraries  
- Dataset providers for enabling fraud detection research  
