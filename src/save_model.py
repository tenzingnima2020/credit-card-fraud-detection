# ============================================================
# CELL 29 — SAVE THE BEST MODEL 
# ============================================================
# We save the trained model to disk using joblib.
# This is what a production pipeline would do — train once, save, then
# load the saved model in an API or app without retraining every time.
# We also save the list of feature columns so the Streamlit app knows
# exactly what input format the model expects.

import joblib

joblib.dump(rf_model, 'fraud_model.pkl')
joblib.dump(X_train.columns.tolist(), 'model_columns.pkl')

print("Model saved as fraud_model.pkl")
print("Column names saved as model_columns.pkl")
print(f"   Total features: {len(X_train.columns)}")