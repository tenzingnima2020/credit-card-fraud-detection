# credit-card-fraud-detection
Machine learning project to detect fraudulent credit card transactions with a Streamlit app.
📌 Project Overview

This project focuses on building a machine learning-based fraud detection system to identify fraudulent credit card transactions.

The system uses real transaction data, applies feature engineering, and trains classification models to detect suspicious activity with high accuracy.

🎯 Objectives
Detect fraudulent transactions using machine learning
Handle imbalanced dataset problem
Improve model performance using feature engineering
Evaluate models using precision, recall, and F1-score
Deploy an interactive Streamlit web application
📊 Dataset
Contains transaction-level data
Includes features like:
Transaction amount
Time
Engineered features (V1–V28)
Highly imbalanced (fraud cases are very rare)
🛠️ Technologies Used
Python
Pandas, NumPy – Data processing
Scikit-learn – Machine learning
XGBoost – Advanced model
Matplotlib / Plotly – Visualization
Streamlit – Web app deployment
⚙️ Project Workflow
1. Data Preprocessing
Handling missing values
Feature scaling
Train-test split
2. Feature Engineering
Created new features like:
is_night (fraud more likely at night)
Transaction patterns
3. Model Building
Logistic Regression
Random Forest
XGBoost (best performing)
4. Model Evaluation
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
📈 Key Results
Improved fraud detection using engineered features
High recall achieved (important for fraud detection)
Balanced performance across precision and recall
🚀 Streamlit Application

The project includes an interactive dashboard where users can:

Input transaction details
Predict whether a transaction is fraud or not
Visualize model performance
Run the app locally:
streamlit run app.py
📁 Project Structure
├── data/
├── models/
├── app.py
├── fraud_model.pkl
├── requirements.txt
└── README.md
🔍 Challenges Faced
Handling highly imbalanced data
Avoiding overfitting
Improving recall without sacrificing precision
💡 Future Improvements
Use deep learning models (Neural Networks)
Deploy on cloud (AWS / Azure)
Real-time fraud detection pipeline
Add more external data sources