import os, joblib, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
np.random.seed(42)
n = 1500

df = pd.DataFrame({
    "square_feet": np.random.randint(500, 4500, size=n),
    "bedrooms": np.random.randint(1, 6, size=n),
    "bathrooms": np.random.randint(1, 5, size=n),
    "property_age_years": np.random.randint(0, 35, size=n),
    "location_tier": np.random.choice([1, 2, 3], size=n, p=[0.35, 0.45, 0.20]),
    "garage_spaces": np.random.choice([0, 1, 2, 3], size=n, p=[0.25, 0.40, 0.25, 0.10])
})

mult = df["location_tier"].map({1: 1.40, 2: 1.10, 3: 0.85})
base = (df["square_feet"] * 60) + (df["bedrooms"] * 18000) + (df["bathrooms"] * 22000) - (df["property_age_years"] * 1100) + (df["garage_spaces"] * 14000) + 60000
df["price"] = np.round((base * mult) + np.random.normal(0, 12000, size=n), 2)
df.to_csv("data/dataset.csv", index=False)
print("[1/3] Dataset written to data/dataset.csv")

X = df.drop(columns=["price"])
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42))
])
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print(f"[2/3] Model trained: R2={r2_score(y_test, y_pred):.4f}, MAE=${mean_absolute_error(y_test, y_pred):,.2f}")

joblib.dump(pipe, "models/model.pkl")
print("[3/3] Exported pipeline to models/model.pkl")