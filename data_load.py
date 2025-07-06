import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import seaborn as sns
from pathlib import Path
import numpy as np

# Base project dir
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / 'model'
MODEL_DIR.mkdir(exist_ok=True)

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Fill missing numeric values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode 'sex'
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

# Drop 'embarked' for simplicity
df.drop(['embarked'], axis=1, inplace=True)

X = df.drop('survived', axis=1)
y = df['survived']

# Add family size feature
X['FamilySize'] = X['sibsp'] + X['parch'] + 1

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Numeric preprocessing
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
    ]
)

# Fit & transform
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save artifacts to model/
joblib.dump(preprocessor, MODEL_DIR / 'preprocessor.joblib')
np.save(MODEL_DIR / 'X_train.npy', X_train_processed)
np.save(MODEL_DIR / 'y_train.npy', y_train.values)
np.save(MODEL_DIR / 'X_test.npy', X_test_processed)
np.save(MODEL_DIR / 'y_test.npy', y_test.values)

print(f"Preprocessor and data saved to {MODEL_DIR}")
