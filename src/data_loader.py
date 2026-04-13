# ============================================================
# CELL 1 — IMPORTING LIBRARIES
# ============================================================
# We import all the libraries we need upfront.
# pandas & numpy → data manipulation
# matplotlib & seaborn → visualizations
# datetime → to dynamically get the current year (fixes the hardcoded 2020 bug)
# warnings → to suppress non-critical output and keep the notebook clean

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)

print("Libraries imported successfully")

# ============================================================
# CELL 2 — LOADING DATASET
# ============================================================
# Load the CSV file into a pandas DataFrame.
# UPDATE the path below to match where your CSV file is saved on your computer.

df = pd.read_csv(r"C:\Users\A S U S\aidi-1204-project\credit-card-fraud-detection\data\credit_card_transactions.csv")

print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ============================================================
# CELL 3 — INITIAL DATA INSPECTION
# ============================================================
# Quick look at the first few rows to understand what the data looks like.
# shape → confirms how big the dataset is
# dtypes → tells us which columns are numeric vs text
# This is the first thing any interviewer expects you to do.

print("Shape:", df.shape)
print("\nColumn Types:")
print(df.dtypes)
df.head()

# ============================================================
# CELL 4 — CHECK FOR MISSING VALUES
# ============================================================
# Checking how many null values exist per column.
# Any column with too many nulls needs to be dropped or imputed.

missing = df.isnull().sum()
print("Missing values per column:")
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found")

