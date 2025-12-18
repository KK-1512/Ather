import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

df = pd.read_csv("ev_range_dataset.csv")

X = df[['battery_level', 'avg_speed', 'temperature', 'riding_mode']]
y = df['estimated_range']

preprocess = ColumnTransformer(
    transformers=[
        ('mode', OneHotEncoder(handle_unknown='ignore'), ['riding_mode'])
    ],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocess', preprocess),
    ('lr', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
print("RMSE:", rmse)
print("R2:", r2)

joblib.dump(model, "ev_model.pkl")
print("saved ev_model.pkl")
