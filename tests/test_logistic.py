import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


from LogisticRegression import LogisticRegression as MyLogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

pipeline_sk = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=100000))
])

pipeline_sk.fit(X_train, y_train)
y_pred_sk = pipeline_sk.predict(X_test)

pipeline_mymodel = Pipeline([
    ('scaler', StandardScaler()),
    ('model', MyLogisticRegression(epochs=100000, lr=0.001))
])

pipeline_mymodel.fit(X_train, y_train)
y_pred_mymodel = pipeline_mymodel.predict(X_test)


print("=== SKLEARN ===")
print('Accuracy:', accuracy_score(y_test, y_pred_sk))
print('Precision:', precision_score(y_test, y_pred_sk))
print('Recall:', recall_score(y_test, y_pred_sk))
print('F1:', f1_score(y_test, y_pred_sk))

print("\n=== MI MODELO ===")
print('Accuracy:', accuracy_score(y_test, y_pred_mymodel))
print('Precision:', precision_score(y_test, y_pred_mymodel))
print('Recall:', recall_score(y_test, y_pred_mymodel))
print('F1:', f1_score(y_test, y_pred_mymodel))