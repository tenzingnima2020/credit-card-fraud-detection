from pathlib import Path

# Project root folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Main folders
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
APP_DIR = BASE_DIR / "app"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Data files
RAW_DATA_PATH = DATA_DIR / "credit_card_transactions.csv"
CLEAN_DATA_PATH = DATA_DIR / "transactions_clean.csv"

# Model files
MODEL_PATH = MODELS_DIR / "fraud_model.pkl"
MODEL_COLUMNS_PATH = MODELS_DIR / "model_columns.pkl"

# Report files
RESULTS_PATH = REPORTS_DIR / "model_results.csv"
FEATURE_IMPORTANCE_PATH = FIGURES_DIR / "feature_importance.png"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrices.png"
ROC_CURVE_PATH = FIGURES_DIR / "roc_curve.png"
PR_CURVE_PATH = FIGURES_DIR / "precision_recall_curve.png"