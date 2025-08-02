import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
from LinearRegression import LinearRegression as MyLinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Cargar datos
cali_data = fetch_california_housing()
X, y = cali_data.data, cali_data.target

# Preprocesamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modelo sklearn
model_sk = LinearRegression()
model_sk.fit(X_train, y_train)
y_pred_sk = model_sk.predict(X_test)

# Tu modelo
model = MyLinearRegression(lr=0.0005, epoch=100000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Métricas
mse_train = np.mean((y_train - model.predict(X_train))**2)
mse_test = np.mean((y_test - y_pred)**2)
mse_test_sk = np.mean((y_test - y_pred_sk)**2)

print("=== SKLEARN ===")
print("w:", model_sk.coef_)
print("b:", model_sk.intercept_)
print("MSE (test):", mse_test_sk)

print("\n=== MI MODELO ===")
print("w:", model.w)
print("b:", model.b)
print("MSE (train):", mse_train)
print("MSE (test):", mse_test)
print("Predicción ejemplo:", y_pred[:5])
print("Valor real:", y_test[:5])