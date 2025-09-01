import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
from LinearRegression import LinearRegression as MyLinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from Lasso import LassoRegression


cali_data = fetch_california_housing()
X, y = cali_data.data, cali_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline_sk = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
pipeline_sk.fit(X_train, y_train)
y_pred_sk = pipeline_sk.predict(X_test)


pipeline_my_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', MyLinearRegression(lr=0.0005, epochs=100000))
])
pipeline_my_model.fit(X_train, y_train)
y_pred_mymodel = pipeline_my_model.predict(X_test)

pipeline_my_lasso = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LassoRegression(lr=0.0005, epochs=100000, alpha=0.001))
])
pipeline_my_lasso.fit(X_train, y_train)
y_pred_my_lasso = pipeline_my_lasso.predict(X_test)

y_pred_train_sk = pipeline_sk.predict(X_train)
mse_train_sk = np.mean((y_train - y_pred_train_sk)**2)
mse_test_sk = np.mean((y_test - y_pred_sk)**2)

y_pred_train_mymodel = pipeline_my_model.predict(X_train)
mse_train = np.mean((y_train - y_pred_train_mymodel)**2)
mse_test = np.mean((y_test - y_pred_mymodel)**2)

y_pred_train_my_lasso = pipeline_my_lasso.predict(X_train)
mse_train_lasso = np.mean((y_train - y_pred_train_my_lasso)**2)
mse_test_lasso = np.mean((y_test - y_pred_my_lasso)**2)


print("=== SKLEARN ===")
print("pesos:", pipeline_sk['model'].coef_)
print("bias:", pipeline_sk['model'].intercept_)
print("MSE (train):", mse_train_sk)
print("MSE (test):", mse_test_sk)


print("\n=== MI MODELO ===")
print("pesos:", pipeline_my_model['model'].weights)
print("bias:", pipeline_my_model['model'].bias)
print("MSE (train):", mse_train)
print("MSE (test):", mse_test)

print("\n=== MI MODELO LASSO ===")
print("pesos:", pipeline_my_lasso['model'].weights)
print("bias:", pipeline_my_lasso['model'].bias)
print("MSE (train):", mse_train_lasso)
print("MSE (test):", mse_test_lasso)