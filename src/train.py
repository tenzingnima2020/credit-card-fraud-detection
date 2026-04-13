# ============================================================
# CELL 17 — TRAIN/TEST SPLIT FIRST  ← FIXED ORDER
# ============================================================
# We split the data BEFORE any resampling. This ensures the test set
# reflects real-world class distribution and hasn't been contaminated
# by upsampled data. random_state=42 ensures reproducibility.

from sklearn.model_selection import train_test_split
from sklearn.utils import resample

X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)  # stratify=y → keeps the same fraud ratio in both train and test

print(f"Train size: {X_train.shape[0]:,}")
print(f"Test size:  {X_test.shape[0]:,}")
print(f"\nFraud % in train: {y_train.mean()*100:.2f}%")
print(f"Fraud % in test:  {y_test.mean()*100:.2f}%")


# ============================================================
# CELL 18 — UPSAMPLE ONLY THE TRAINING DATA  ← FIXED
# ============================================================
# Now we balance the training set by oversampling the minority (fraud) class.
# We use resample() to create synthetic copies of fraud cases until it matches
# the number of legitimate transactions. This is done ONLY on training data.
# The test set stays untouched so our evaluation is realistic.

train_df = pd.concat([X_train, y_train], axis=1)

majority = train_df[train_df['is_fraud'] == 0]
minority = train_df[train_df['is_fraud'] == 1]

minority_upsampled = resample(minority,
                               replace=True,
                               n_samples=len(majority),
                               random_state=42)

train_balanced = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=42)

X_train = train_balanced.drop('is_fraud', axis=1)
y_train = train_balanced['is_fraud']

print(f"After balancing — Train size: {X_train.shape[0]:,}")
print(f"Fraud cases in training: {y_train.sum():,}")
print(f"Legit cases in training: {(y_train==0).sum():,}")


# ============================================================
# CELL 19 — TRAIN RANDOM FOREST
# ============================================================
# Random Forest is an ensemble of decision trees — it votes across 100 trees
# to make a prediction. It handles non-linear patterns well and gives us
# feature importances. max_depth=10 prevents the trees from overfitting.

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    n_jobs=-1,
    random_state=42
)
rf_model.fit(X_train, y_train)
print("Random Forest trained")


# ============================================================
# CELL 20 — TRAIN LOGISTIC REGRESSION
# ============================================================
# Logistic Regression is a simpler, interpretable linear model.
# It works well as a baseline to compare against more complex models.
# class_weight='balanced' helps it handle the remaining imbalance.

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(
    max_iter=500,
    solver='liblinear',
    class_weight='balanced',
    random_state=42
)
lr_model.fit(X_train, y_train)
print("Logistic Regression trained")


# ============================================================
# CELL 21 — TRAIN XGBOOST  ← NEW MODEL
# ============================================================
# XGBoost is a powerful gradient boosting algorithm — often top-performing
# on tabular data. scale_pos_weight compensates for class imbalance by
# telling XGBoost how much more to weight the minority (fraud) class.
# Adding XGBoost shows you know multiple model families, not just sklearn.

from xgboost import XGBClassifier

scale = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
xgb_model.fit(X_train, y_train)
print("XGBoost trained")