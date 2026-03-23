import pandas as pd
import joblib
import argparse
import os

def predict(input_path, output_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    original_df = df.copy()
    
    # Preprocess
    for col in ['customer_id', 'name']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            
    # Load artifacts
    print("Loading model artifacts...")
    try:
        model = joblib.load('models/xgb_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_cols = joblib.load('models/feature_cols.pkl')
    except Exception as e:
        print("Failed to load models. Did you run train.py first?")
        return
        
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            val = df[col].median()
            df[col] = df[col].fillna(val if pd.notna(val) else 0)
        else:
            val = df[col].mode()
            df[col] = df[col].fillna(val[0] if len(val) > 0 else 'Unknown')
    df = df.fillna(0)
                
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        
    # Align features
    df = df.reindex(columns=feature_cols, fill_value=0)
    
    # Scale
    X_scaled = scaler.transform(df)
    
    print("Running inference...")
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    original_df['prediction'] = predictions
    original_df['PD_Probability_of_Default'] = probabilities
    
    # Calculate EAD, LGD, Expected Loss if columns exist
    if 'credit_limit' in original_df.columns and 'credit_limit_used(%)' in original_df.columns:
        current_balance = original_df['credit_limit'] * (original_df['credit_limit_used(%)'] / 100.0)
        undrawn_amount = original_df['credit_limit'] - current_balance
        ccf = 0.75 # 75% credit conversion factor
        original_df['EAD_Exposure_At_Default'] = current_balance + (ccf * undrawn_amount)
        
        original_df['LGD_Loss_Given_Default'] = 0.75 # Assumed 75%
        original_df['EL_Expected_Loss'] = original_df['PD_Probability_of_Default'] * original_df['LGD_Loss_Given_Default'] * original_df['EAD_Exposure_At_Default']
    else:
        # Fallback if credit features are missing
        original_df['LGD_Loss_Given_Default'] = 0.75
        
    original_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test.csv", help="Path to input cvs")
    parser.add_argument("--output", default="predictions.csv", help="Path to output csv")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Input file {args.input} does not exist.")
    else:
        predict(args.input, args.output)
