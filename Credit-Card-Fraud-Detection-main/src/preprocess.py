import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(filepath):
    """Load and return the raw dataset."""
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
    return df

def clean_data(df):
    """Remove duplicates and handle nulls."""
    initial_shape = df.shape
    df = df.drop_duplicates()
    df = df.dropna()
    print(f"Removed {initial_shape[0] - df.shape[0]} duplicate/null rows")
    return df

def engineer_features(df):
    """Create new features from existing ones."""
    # Hour of day from Time (seconds since first transaction)
    df = df.copy()
    df['Hour'] = (df['Time'] // 3600) % 24
    
    # Log transform of Amount (reduces skew)
    df['Amount_log'] = np.log1p(df['Amount'])
    
    # Amount bins
    df['Amount_bin'] = pd.cut(
        df['Amount'],
        bins=[0, 10, 50, 200, 500, np.inf],
        labels=['tiny', 'small', 'medium', 'large', 'huge']
    )
    df['Amount_bin'] = df['Amount_bin'].astype(str)
    
    # Drop original Time and Amount (replaced by engineered versions)
    df = df.drop(['Time', 'Amount'], axis=1)
    df = df.drop(['Amount_bin'], axis=1)  # drop string column for now
    
    return df

def scale_features(df):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )
    
    return X_scaled, y, scaler

def split_and_oversample(X, y, test_size=0.2, random_state=42):
    """Split data and apply SMOTE only on training set."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train set: {X_train.shape}, Fraud: {y_train.sum()}")
    print(f"Test set: {X_test.shape}, Fraud: {y_test.sum()}")
    
    # Apply SMOTE only on training data — NEVER on test data
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    print(f"After SMOTE - Train: {X_train_res.shape}, Fraud: {y_train_res.sum()}")
    
    return X_train_res, X_test, y_train_res, y_test