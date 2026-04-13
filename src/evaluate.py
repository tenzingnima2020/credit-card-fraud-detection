# ============================================================
# CELL 22 — PREDICTIONS FOR ALL THREE MODELS
# ============================================================
# Generate predictions and probability scores for each model.
# predict() → hard labels (0 or 1)
# predict_proba()[:,1] → probability of being fraud (used for ROC/AUC)

y_pred_rf  = rf_model.predict(X_test)
y_pred_lr  = lr_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

y_prob_rf  = rf_model.predict_proba(X_test)[:, 1]
y_prob_lr  = lr_model.predict_proba(X_test)[:, 1]
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("Predictions generated for all 3 models")


# ============================================================
# CELL 23 — MODEL COMPARISON TABLE  ← NEW
# ============================================================
# This summary table is the most interview-friendly thing in the whole notebook.
# It lets you immediately compare all three models on the metrics that matter.
# F1 and AUC are far more meaningful than accuracy for fraud detection.

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)

results = {
    'Model': ['Random Forest', 'Logistic Regression', 'XGBoost'],
    'Accuracy':  [accuracy_score(y_test, y_pred_rf),
                  accuracy_score(y_test, y_pred_lr),
                  accuracy_score(y_test, y_pred_xgb)],
    'Precision': [precision_score(y_test, y_pred_rf),
                  precision_score(y_test, y_pred_lr),
                  precision_score(y_test, y_pred_xgb)],
    'Recall':    [recall_score(y_test, y_pred_rf),
                  recall_score(y_test, y_pred_lr),
                  recall_score(y_test, y_pred_xgb)],
    'F1 Score':  [f1_score(y_test, y_pred_rf),
                  f1_score(y_test, y_pred_lr),
                  f1_score(y_test, y_pred_xgb)],
    'ROC-AUC':   [roc_auc_score(y_test, y_prob_rf),
                  roc_auc_score(y_test, y_prob_lr),
                  roc_auc_score(y_test, y_prob_xgb)],
}

results_df = pd.DataFrame(results).set_index('Model').round(4)
print("=== Model Comparison ===")
results_df


# ============================================================
# CELL 24 — CONFUSION MATRICES FOR ALL THREE MODELS  ← UPGRADED
# ============================================================
# A confusion matrix shows True Positives, False Positives, etc.
# In fraud detection, False Negatives (missed fraud) are the most costly.
# Showing all three side by side makes for a clean, professional comparison.

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

models_info = [
    ('Random Forest',       y_pred_rf),
    ('Logistic Regression', y_pred_lr),
    ('XGBoost',             y_pred_xgb),
]

for ax, (name, preds) in zip(axes, models_info):
    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(cm, display_labels=['Legit', 'Fraud']).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(name, fontsize=12, fontweight='bold')

plt.suptitle('Confusion Matrices — All Models', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# ============================================================
# CELL 25 — ROC CURVES  ← NEW
# ============================================================
# The ROC Curve plots True Positive Rate vs False Positive Rate at every
# classification threshold. AUC (Area Under Curve) summarizes this in one number.
# A perfect model has AUC = 1.0. A random guess gives AUC = 0.5.
# This is the gold standard chart for binary classifiers in interviews.

from sklearn.metrics import roc_curve

plt.figure(figsize=(9, 6))

for name, probs in [('Random Forest', y_prob_rf),
                     ('Logistic Regression', y_prob_lr),
                     ('XGBoost', y_prob_xgb)]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Guess')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves — All Models', fontsize=13, fontweight='bold')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# ============================================================
# CELL 26 — PRECISION-RECALL CURVES  ← NEW
# ============================================================
# For highly imbalanced datasets, Precision-Recall curves are even more
# informative than ROC curves. They focus specifically on how well the model
# performs on the minority (fraud) class — which is what we actually care about.
# A high Average Precision score means the model catches fraud with few false alarms.

from sklearn.metrics import precision_recall_curve, average_precision_score

plt.figure(figsize=(9, 6))

for name, probs in [('Random Forest', y_prob_rf),
                     ('Logistic Regression', y_prob_lr),
                     ('XGBoost', y_prob_xgb)]:
    prec, rec, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    plt.plot(rec, prec, linewidth=2, label=f'{name} (AP = {ap:.3f})')

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves — All Models', fontsize=13, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# CELL 27 — DETAILED CLASSIFICATION REPORTS
# ============================================================
# Full breakdown of Precision, Recall, F1 per class for each model.
# Helps explain in an interview *which model is best and why*.

from sklearn.metrics import classification_report

for name, preds in [('Random Forest', y_pred_rf),
                     ('Logistic Regression', y_pred_lr),
                     ('XGBoost', y_pred_xgb)]:
    print(f"{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(classification_report(y_test, preds, target_names=['Legitimate', 'Fraud']))
    print()


# ============================================================
# CELL 28 — FEATURE IMPORTANCE (RANDOM FOREST) 
# ============================================================
# Random Forest automatically calculates how useful each feature was
# for making predictions. Higher importance = more useful.
# This chart is a great talking point in interviews — you can explain
# *why* certain features (like amount, distance) matter more than others.

feat_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
top_features = feat_importances.nlargest(15)

plt.figure(figsize=(10, 6))
bars = plt.barh(top_features.index[::-1], top_features.values[::-1],
                color=sns.color_palette('Blues_d', 15), edgecolor='black')
plt.xlabel('Feature Importance Score')
plt.title('Top 15 Most Important Features — Random Forest', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nTop 15 Features:")
print(top_features.round(4).to_string())
