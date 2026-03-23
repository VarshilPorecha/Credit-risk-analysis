# Quantitative Credit Risk Analysis (IRB Framework)

This project builds an end-to-end machine learning pipeline to predict credit defaults using financial customer data. It implements a Basel Internal Ratings-Based (IRB) approach by estimating Probability of Default (PD), Exposure at Default (EAD), and Loss Given Default (LGD) to calculate Expected Financial Loss.

## Features & Quantitative Metrics
- **Probability of Default (PD)**: Predicted using an optimized XGBoost classifier.
- **Exposure at Default (EAD)**: Derived dynamically using current balance, undrawn limits, and a Credit Conversion Factor (CCF ~ 75%).
- **Loss Given Default (LGD)**: Assumed regulatory standard (75%) for unsecured consumer credit.
- **Expected Loss (EL)**: Calculated at the portfolio and individual level to estimate direct monetary value at risk (`EL = PD * LGD * EAD`).

## Structure
- `data/` : Original datasets (e.g., `train.csv` and `test.csv`)
- `notebooks/credit_risk_analysis.ipynb` : The main Jupyter Notebook containing EDA, Feature Engineering, Model Testing, Visualizations, and conclusions.
- `src/train.py` : Script to automate model training of the best model (XGBoost) and save the model artifact.
- `src/predict.py` : Script to load the saved model and run inference on new tabular data.
- `models/` : Serialized models are saved here.

## Setup Instructions
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Running the notebook:
   ```bash
   jupyter notebook notebooks/credit_risk_analysis.ipynb
   ```
3. Training the production model:
   ```bash
   python src/train.py
   ```
   This will train the XGBoost model and save it to `models/xgb_model.pkl`.

4. Running inference:
   ```bash
   python src/predict.py --input test.csv --output predictions.csv
   ```
