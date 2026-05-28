"""
Credit Card Fraud Detection System
Run this file to execute the complete pipeline.
"""

import os
import warnings
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

from src.preprocess import (
    load_data, clean_data, engineer_features,
    scale_features, split_and_oversample
)
from src.train import (
    train_random_forest, train_xgboost,
    evaluate_model, save_model
)
from src.visualize import (
    plot_class_distribution, plot_amount_distribution,
    plot_confusion_matrix, plot_roc_curve,
    plot_feature_importance, plot_hourly_fraud
)
from src.predict import predict_transaction, batch_predict

def main():
    print("\n" + "="*60)
    print("  CREDIT CARD FRAUD DETECTION SYSTEM")
    print("  Starting pipeline...")
    print("="*60 + "\n")

    # ── Phase 1: Load ──────────────────────────────────────────
    print("📂 PHASE 1: Loading Data")
    df = load_data('data/creditcard.csv')

    # ── Phase 2: Clean ─────────────────────────────────────────
    print("\n🧹 PHASE 2: Cleaning Data")
    df = clean_data(df)

    # ── Phase 3: EDA Visualization ─────────────────────────────
    print("\n📊 PHASE 3: Generating EDA Charts")
    plot_class_distribution(df['Class'])
    plot_amount_distribution(df)

    # ── Phase 4: Feature Engineering ───────────────────────────
    print("\n⚙️  PHASE 4: Feature Engineering")
    df = engineer_features(df)
    plot_hourly_fraud(df)

    # ── Phase 5: Scale + Split + SMOTE ─────────────────────────
    print("\n⚖️  PHASE 5: Scaling and Oversampling")
    X, y, scaler = scale_features(df)
    X_train, X_test, y_train, y_test = split_and_oversample(X, y)

    # ── Phase 6: Train Models ───────────────────────────────────
    print("\n🤖 PHASE 6: Training Models")
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    # ── Phase 7: Evaluate ───────────────────────────────────────
    print("\n📈 PHASE 7: Evaluating Models")
    rf_results  = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # ── Phase 8: Save Charts ────────────────────────────────────
    print("\n💾 PHASE 8: Saving Evaluation Charts")
    plot_confusion_matrix(rf_results['confusion_matrix'])
    plot_roc_curve(y_test, rf_results['y_prob'], xgb_results['y_prob'])
    plot_feature_importance(xgb_model, X.columns.tolist())

    # ── Phase 9: Save Models ────────────────────────────────────
    print("\n💾 PHASE 9: Saving Models")
    save_model(rf_model,  'models/random_forest.pkl')
    save_model(xgb_model, 'models/xgboost_model.pkl')
    save_model(scaler,    'models/scaler.pkl')

    # ── Phase 10: Simulate Live Prediction ─────────────────────
    print("\n🚨 PHASE 10: Simulating Live Transaction")
    
    # Build a test transaction from the test set
    test_transaction = X_test.iloc[0].to_dict()
    
    predict_transaction(xgb_model, scaler, 
                        {**test_transaction, 'Time': 45000, 'Amount': 250.0})
    
    # Batch prediction demo
    print("\n📦 Batch Prediction on Test Set:")
    batch_predict(xgb_model, scaler, X_test, y_test)

    print("\n✅ Pipeline complete. Check outputs/ for all charts.")
    print("🚀 Run: streamlit run app/streamlit_app.py\n")

if __name__ == '__main__':
    main()