
# Mi Propia Regresión Lineal y Logística

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-green)](https://www.python.org/downloads/)

Este proyecto implementa algoritmos de **Regresión Lineal** y **Regresión Logística** desde cero utilizando NumPy, con el objetivo de entender los conceptos fundamentales detrás de estos métodos básicos de Machine Learning.

## 🎯 Descripción del Proyecto

Ambas implementaciones buscan ser educativas, demostrando:
- Fundamentos matemáticos claros.
- Optimización mediante descenso de gradiente.
- Funciones de pérdida y métricas de evaluación.
- Comparación directa con implementaciones de scikit-learn.

## 📁 Estructura del Proyecto

```
MyOwnLinearRegression/
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── LinearRegression.py      # Implementación desde cero de Regresión Lineal
│   └── LogisticRegression.py    # Implementación desde cero de Regresión Logística
└── tests/
    ├── test_linear.py           # Pruebas para regresión lineal
    └── test_logistic.py         # Pruebas para regresión logística
```

## 🚀 Características

### Regresión Lineal
- Regresión lineal multivariante.
- Función de pérdida: Error Cuadrático Medio (MSE).
- Optimización mediante descenso de gradiente.
- Validado con el dataset **California Housing**.

### Regresión Logística
- Clasificación binaria.
- Función de activación: Sigmoide.
- Función de pérdida: Log Loss (Cross-Entropy).
- Validado con el dataset **Breast Cancer Wisconsin**.

## 📦 Instalación

Clona el repositorio:
```bash
git clone <url-del-repositorio>
cd MyOwnLinearRegression
```

Instala dependencias:
```bash
pip install -r requirements.txt
```

## 🔧 Uso Rápido

**Nota**: Debes separar previamente tus datos en conjuntos de entrenamiento y prueba (`train_test_split`).

### Ejemplo: Regresión Lineal
```python
from src.LinearRegression import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression(epoch=1000, lr=0.001)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Ejemplo: Regresión Logística
```python
from src.LogisticRegression import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(epoch=10000, lr=0.0001)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## 🧪 Pruebas

Ejecuta scripts para comparar tus modelos con scikit-learn:

```bash
python tests/test_linear.py
python tests/test_logistic.py
```

Resultados ejemplo:

```
=== SKLEARN (Logistic) ===
Accuracy: 0.9860
F1 Score: 0.9888

=== MI MODELO (Logistic) ===
Accuracy: 0.9790
F1 Score: 0.9832
```

## 📊 Resultados Obtenidos

### Regresión Lineal (California Housing)
- **MSE** en conjunto de prueba comparable a scikit-learn.

### Regresión Logística (Breast Cancer Wisconsin)
- **Accuracy** superior al 97%.
- **Precision**, **Recall** y **F1-Score** similares a scikit-learn.

## 🧮 Fundamentos Matemáticos

| Modelo              | Hipótesis                      | Función de Pérdida                                 |
|---------------------|--------------------------------|-----------------------------------------------------|
| Regresión Lineal    | `y = Xw + b`                   | ![MSE](https://latex.codecogs.com/svg.image?MSE=\frac{1}{n}\sum\left(y-\hat{y}\right)^{2}) |
| Regresión Logística | `y = sigmoid(Xw + b)`          | ![LogLoss](https://latex.codecogs.com/svg.image?LogLoss=-\frac{1}{n}\sum\left[y\log(\hat{y})\,+\,(1-y)\log(1-\hat{y})\right]) |


- **Descenso de Gradiente:** Actualización iterativa usando derivadas parciales.

## 🛠️ Dependencias
- `numpy`: Cálculos numéricos.
- `scikit-learn`: Datasets y comparación.

Instala todas con `pip install -r requirements.txt`.

## 📚 Referencias y Recursos
- Libro: **Mathematics for Machine Learning** (Deisenroth & Faisal)
- Curso: **Mathematics for Machine Learning** (Imperial College London - Coursera)
- Documentación oficial de [scikit-learn](https://scikit-learn.org/stable/)

## 🤝 Cómo Contribuir
¡Tu contribución es bienvenida!
- Abre un Issue con mejoras.
- Envía un Pull Request.
- Mejora documentación y agrega ejemplos.

## 📄 Licencia
[MIT](LICENSE)

---

**Nota Final:** Este proyecto es educativo. Para producción utiliza librerías profesionales como **scikit-learn**, **TensorFlow**, o **PyTorch**.
