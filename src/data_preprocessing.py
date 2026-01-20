import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data(input_path, output_data_path, output_model_path):
  # Load Data
  df = pd.read_csv(input_path)
  
  # Cleaning & Encoding (Sama seperti di notebook)
  df = df.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
  df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})
  
  X = df.drop('Machine failure', axis=1)
  y = df['Machine failure']
  
  # Split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
  
  # Scaling
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  
  # Save Scaler (sesuai struktur models/preprocessing.pkl)
  joblib.dump(scaler, f"{output_model_path}/preprocessing.pkl")
  
  # Save Split Data
  data = {'X_train': X_train_scaled, 'X_test': X_test_scaled, 'y_train': y_train, 'y_test': y_test}
  joblib.dump(data, f"{output_data_path}/split_data.pkl")
  
  print("Preprocessing Selesai.")

if __name__ == "__main__":
  # Contoh cara jalanin script
  preprocess_data('data/raw/ai4i2020.csv', 'data/processed', 'models')