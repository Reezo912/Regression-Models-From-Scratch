from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression




from src.LogisticRegression import LogisticRegression2

# Cargar datos
data = load_breast_cancer()
X, y = data.data, data.target

# Escalar features para que el aprendizaje sea más estable (muy recomendable)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# sklearn
model_sk = LogisticRegression(max_iter=100000, solver='liblinear')
model_sk.fit(X_train, y_train)
y_pred_sk = model_sk.predict(X_test)

# Tu modelo
model = LogisticRegression2(epoch=100000, lr=0.001)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Métricas
print("=== SKLEARN ===")
print('Accuracy:', accuracy_score(y_test, y_pred_sk))
print('Precision:', precision_score(y_test, y_pred_sk))
print('Recall:', recall_score(y_test, y_pred_sk))
print('F1:', f1_score(y_test, y_pred_sk))

print("\n=== TU MODELO ===")
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1:', f1_score(y_test, y_pred))