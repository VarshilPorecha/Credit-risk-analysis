import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

def train_model():
    print("Loading data...")
    # Load data
    data_path = 'train.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Please place train.csv in root.")
    
    df = pd.read_csv(data_path)
    
    print("Preprocessing data...")
    # Clean identifying columns
    for col in ['customer_id', 'name']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            
    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            val = df[col].median()
            df[col] = df[col].fillna(val if pd.notna(val) else 0)
        else:
            val = df[col].mode()
            df[col] = df[col].fillna(val[0] if len(val) > 0 else 'Unknown')
    df = df.fillna(0)
                
    # Feature Engineering (Dummy Encoding)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        
    print("Splitting data and handling imbalance...")
    X = df.drop('credit_card_default', axis=1)
    y = df['credit_card_default']
    
    # Save the feature columns to ensure consistency in prediction
    feature_cols = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training XGBoost Model...")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train_res)
    
    score = model.score(X_test_scaled, y_test)
    print(f"Model trained successfully. Test Accuracy: {score:.4f}")
    
    print("Saving model and scaler...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgb_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')
    print("Saved to models directory.")

if __name__ == "__main__":
    train_model()
