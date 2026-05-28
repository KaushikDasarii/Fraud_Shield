import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
from xgboost import XGBClassifier

def train_random_forest(X_train, y_train):
    """Train Random Forest classifier."""
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        class_weight='balanced',   # additional help for imbalance
        random_state=42,
        n_jobs=-1                  # use all CPU cores
    )
    rf.fit(X_train, y_train)
    print("Random Forest training complete.")
    return rf

def train_xgboost(X_train, y_train):
    """Train XGBoost classifier."""
    print("Training XGBoost...")
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos,  # handles imbalance
        use_label_encoder=False,
        eval_metric='aucpr',         # area under precision-recall curve
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    print("XGBoost training complete.")
    return xgb

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Print full evaluation report."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"\n{'='*50}")
    print(f"  {model_name} Evaluation")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))
    
    roc_auc = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    
    print(f"ROC-AUC Score:          {roc_auc:.4f}")
    print(f"Avg Precision (PR-AUC): {avg_prec:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': avg_prec,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

def save_model(model, filepath):
    """Save model to disk."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Load model from disk."""
    return joblib.load(filepath)