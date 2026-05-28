import numpy as np
import pandas as pd
import joblib
from datetime import datetime

def predict_transaction(model, scaler, transaction_data):
    """
    Predict whether a single transaction is fraudulent.
    
    Args:
        model: Trained classifier
        scaler: Fitted StandardScaler
        transaction_data: dict with keys matching training features
    
    Returns:
        dict with prediction, probability, and alert message
    """
    df = pd.DataFrame([transaction_data])
    
    # Apply same feature engineering
    df['Amount_log'] = np.log1p(df.get('Amount', 0))
    df['Hour'] = (df.get('Time', 0) // 3600) % 24
    
    # Drop raw features
    df = df.drop(['Time', 'Amount'], axis=1, errors='ignore')
    
    # Scale
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=df.columns
    )
    
    # Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]
    
    # Generate alert
    result = {
        'prediction': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
        'fraud_probability': float(probability),
        'risk_level': get_risk_level(probability),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'recommended_action': get_action(prediction, probability)
    }
    
    # Print colored alert
    print_alert(result)
    
    return result

def get_risk_level(prob):
    if prob >= 0.8:
        return 'CRITICAL'
    elif prob >= 0.5:
        return 'HIGH'
    elif prob >= 0.3:
        return 'MEDIUM'
    else:
        return 'LOW'

def get_action(prediction, prob):
    if prediction == 1 and prob >= 0.8:
        return 'BLOCK TRANSACTION — High confidence fraud'
    elif prediction == 1 and prob >= 0.5:
        return 'HOLD FOR REVIEW — Medium confidence fraud'
    elif prediction == 0 and prob >= 0.3:
        return 'FLAG FOR MONITORING — Suspicious pattern'
    else:
        return 'APPROVE — Transaction appears legitimate'

def print_alert(result):
    """Print a formatted terminal alert."""
    border = '=' * 55
    print(f"\n{border}")
    if result['prediction'] == 'FRAUD':
        print(f"  ⚠️  FRAUD ALERT TRIGGERED")
        print(f"  Risk Level    : {result['risk_level']}")
    else:
        print(f"  ✅  TRANSACTION CLEARED")
        print(f"  Risk Level    : {result['risk_level']}")
    print(f"  Prediction    : {result['prediction']}")
    print(f"  Fraud Prob    : {result['fraud_probability']:.2%}")
    print(f"  Action        : {result['recommended_action']}")
    print(f"  Timestamp     : {result['timestamp']}")
    print(f"{border}\n")

def batch_predict(model, scaler, X_test, y_test=None):
    """Run predictions on a batch and return summary report."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    print("\n📊 BATCH PREDICTION SUMMARY")
    print(f"{'='*40}")
    print(f"Total transactions:      {len(y_pred):,}")
    print(f"Predicted fraud:         {y_pred.sum():,}")
    print(f"Predicted legitimate:    {(y_pred == 0).sum():,}")
    print(f"Avg fraud probability:   {y_prob.mean():.4f}")
    
    if y_test is not None:
        actual_fraud = y_test.sum()
        print(f"\nActual fraud cases:      {actual_fraud:,}")
        correctly_caught = ((y_pred == 1) & (y_test == 1)).sum()
        print(f"Correctly caught:        {correctly_caught:,}")
        missed = ((y_pred == 0) & (y_test == 1)).sum()
        print(f"Missed fraud cases:      {missed:,}")
    
    return y_pred, y_prob