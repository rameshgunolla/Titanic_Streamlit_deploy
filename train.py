from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
from pathlib import Path

def train_and_save_model(X_train, y_train, output_dir):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, output_dir / 'model.joblib')
    print(f"Model saved to {output_dir / 'model.joblib'}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent
    MODEL_DIR = BASE_DIR / 'model'

    X_train = np.load(MODEL_DIR / 'X_train.npy')
    y_train = np.load(MODEL_DIR / 'y_train.npy')

    train_and_save_model(X_train, y_train, MODEL_DIR)
