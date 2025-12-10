import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and clean data
df = pd.read_csv("train.csv")

# Drop columns with >50% missing values
df = df.loc[:, df.isnull().mean() < 0.5]

# Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Drop ID column if it exists
if 'Id' in df.columns:
    df.drop('Id', axis=1, inplace=True)

# Split target and features
y = df['SalePrice']
X = df.drop('SalePrice', axis=1)

# One-hot encode all categoricals
X_encoded = pd.get_dummies(X, drop_first=False)
X_encoded.to_csv("model_input_template.csv", index=False)

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and scaler
pickle.dump(model, open("linear_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Model, Scaler, and Template saved successfully!")
