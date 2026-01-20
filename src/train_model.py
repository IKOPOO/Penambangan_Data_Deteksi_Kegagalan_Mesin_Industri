import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train(data_path, model_output_path):
  # Load Data
  data = joblib.load(f"{data_path}/split_data.pkl")
  X_train, y_train = data['X_train'], data['y_train']
  
  # Init Model (Contoh pakai RF)
  model = RandomForestClassifier(class_weight='balanced', random_state=42)
  model.fit(X_train, y_train)
  
  # Save Model
  joblib.dump(model, f"{model_output_path}/best_model.pkl")
  print("Training Selesai. Model disimpan.")

if __name__ == "__main__":
  train('data/processed', 'models')