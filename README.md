# Credit Risk Analysis

This project builds an end-to-end machine learning pipeline to predict credit defaults using financial customer data. 

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
