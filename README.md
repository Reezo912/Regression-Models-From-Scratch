# Mi Propia Regresión Lineal y Logística

Este proyecto implementa algoritmos de **Regresión Lineal** y **Regresión Logística** desde cero usando NumPy. El objetivo es entender los conceptos fundamentales detrás de estos algoritmos básicos de machine learning construyéndolos manualmente.

## 🎯 Descripción del Proyecto

Ambas implementaciones están diseñadas para ser educativas y demuestran:
- **Fundamentos matemáticos** de los algoritmos de regresión
- **Optimización por descenso de gradiente**
- **Cálculos de funciones de pérdida**
- **Comparación de rendimiento** con implementaciones de scikit-learn

## 📁 Estructura del Proyecto

```
MyOwnLinearRegression/
├── README.md
├── requirements.txt
├── src/
│   ├── LinearRegression.py      # Implementación de Regresión Lineal
│   └── LogisticRegression.py    # Implementación de Regresión Logística
└── tests/
    ├── test_linear.py           # Prueba para regresión lineal
    └── test_logistic.py         # Prueba para regresión logística
```

## 🚀 Características

### Regresión Lineal
- **Regresión lineal multivariante** implementada
- **Función de pérdida Error Cuadrático Medio (MSE)**
- **Optimización por descenso de gradiente**
- **Pruebas con dataset California Housing**
- **Comparación de rendimiento** con scikit-learn

### Regresión Logística
- **Clasificación binaria** implementada
- **Función de activación sigmoidea**
- **Optimización de pérdida logarítmica (cross-entropy)**
- **Pruebas con dataset Breast Cancer**
- **Evaluación de métricas de clasificación**

## 📦 Instalación

1. Clona el repositorio:
```bash
git clone <url-del-repositorio>
cd MyOwnLinearRegression
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## 🔧 Uso

### Regresión Lineal

```python
from src.LinearRegression import LinearRegression
from sklearn.datasets import fetch_california_housing

# Cargar y preparar datos
data = fetch_california_housing()
X, y = data.data, data.target

# Crear y entrenar modelo
model = LinearRegression(epoch=1000, lr=0.001)
model.fit(X_train, y_train)

# Hacer predicciones
predictions = model.predict(X_test)
```

### Regresión Logística

```python
from src.LogisticRegression import LogisticRegression
from sklearn.datasets import load_breast_cancer

# Cargar y preparar datos
data = load_breast_cancer()
X, y = data.data, data.target

# Crear y entrenar modelo
model = LogisticRegression(epoch=10000, lr=0.0001)
model.fit(X_train, y_train)

# Hacer predicciones
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## 🧪 Pruebas

Ejecuta los scripts de prueba para comparar tus implementaciones con scikit-learn:

```bash
python tests/test_linear.py
python tests/test_logistic.py
```

Esto mostrará las métricas de rendimiento tanto para tu implementación como para la versión de scikit-learn.

## 📊 Resultados

### Regresión Lineal
- **Dataset**: California Housing
- **Características**: 8 características numéricas
- **Objetivo**: Valor mediano de la casa
- **Rendimiento**: Comparable a la implementación de scikit-learn

### Regresión Logística
- **Dataset**: Breast Cancer Wisconsin
- **Características**: 30 características numéricas
- **Objetivo**: Clasificación binaria (maligno/benigno)
- **Métricas**: Accuracy, Precision, Recall, F1-Score

## 🧮 Antecedentes Matemáticos

### Regresión Lineal
- **Hipótesis**: `y = X @ w + b`
- **Función de Pérdida**: `MSE = (1/n) * Σ(y_true - y_pred)²`
- **Descenso de Gradiente**: Actualiza pesos usando derivadas parciales

### Regresión Logística
- **Hipótesis**: `y = sigmoid(X @ w + b)`
- **Activación**: `sigmoid(z) = 1/(1 + e^(-z))`
- **Función de Pérdida**: `Log Loss = -Σ(y_true * log(y_pred) + (1-y_true) * log(1-y_pred))`

## 🛠️ Dependencias

- `numpy`: Cálculos numéricos
- `scikit-learn`: Carga de datasets y comparación

## 📚 Recursos de Aprendizaje

Este proyecto fue inspirado por:
- **Mathematics for Machine Learning** de Marc Peter Deisenroth y Aldo Faisal
- **Curso "Mathematics for Machine Learning" del Imperial College London en Coursera**
- **Documentación de Scikit-learn** para comparación de implementaciones

## 🤝 Contribuciones

Siéntete libre de contribuir:
- Agregando nuevas características
- Mejorando la documentación
- Optimizando algoritmos
- Agregando más casos de prueba

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la Licencia MIT.

---

**Nota**: Este proyecto es principalmente educativo. Para uso en producción, considera usar bibliotecas establecidas como scikit-learn, TensorFlow, o PyTorch.