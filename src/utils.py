import joblib
import pandas as pd

def load_data(path):
  return pd.read_csv(path)

def save_model(model, path):
  joblib.dump(model, path)
  print(f"Model saved to {path}")

def load_model(path):
  return joblib.load(path)