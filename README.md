
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
git clone https://github.com/Reezo912/Regression-Models-From-Scratch
cd Regression-Models-From-Scratch
```

Instala dependencias:
```bash
pip install -r requirements.txt
```


## 🧪 Pruebas

Ejecuta scripts para comparar los modelos con scikit-learn:

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

| Modelo              | Hipótesis                                                                                                                                     | Función de Pérdida                                                                                                                 |
|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| Regresión Lineal    | ![LinearHypothesis](https://latex.codecogs.com/png.image?\dpi{110}\bg{white}y=Xw+b)                                                          | ![MSE](https://latex.codecogs.com/png.image?\dpi{110}\bg{white}MSE=\frac{1}{n}\sum\left(y-\hat{y}\right)^{2})                     |
| Regresión Logística | ![LogisticHypothesis](https://latex.codecogs.com/png.image?\dpi{110}\bg{white}y=sigmoid(Xw+b))                                               | ![LogLoss](https://latex.codecogs.com/png.image?\dpi{110}\bg{white}LogLoss=-\frac{1}{n}\sum\left[y\log(\hat{y})+(1-y)\log(1-\hat{y})\right]) |


- **Descenso de Gradiente:** Actualización iterativa usando derivadas parciales.

## 🛠️ Dependencias
- `numpy`: Cálculos numéricos.
- `scikit-learn`: Datasets y comparación.

Instala todas con `pip install -r requirements.txt`.

## 🔧 Solución de Problemas

### Error: "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### Error: "ImportError"
Asegúrate de estar en el directorio raíz del proyecto.

### Error: "ValueError: Found input variables with inconsistent numbers of samples"
Verifica que X_train y y_train tengan el mismo número de muestras.

### Error: "ConvergenceWarning"
Aumenta el número de epochs o ajusta el learning rate.

## 📚 Referencias y Recursos
- Libro: **Mathematics for Machine Learning** (Deisenroth & Faisal)
- Curso: **Mathematics for Machine Learning** (Imperial College London - Coursera)
- Documentación oficial de [scikit-learn](https://scikit-learn.org/stable/)

## 🚀 Roadmap

### Próximas Características
- [ ] Implementación de Regularización (Ridge/Lasso)
- [ ] Regresión Logística Multiclase
- [ ] Validación Cruzada
- [ ] Visualizaciones de resultados
- [ ] Optimizadores alternativos (Adam, SGD)

### Mejoras Técnicas
- [ ] Paralelización del entrenamiento
- [ ] Early stopping
- [ ] Learning rate scheduling
- [ ] Batch gradient descent

## 🤝 Cómo Contribuir
¡Tu contribución es bienvenida!
- Abre un Issue con mejoras.
- Envía un Pull Request.
- Mejora documentación y agrega ejemplos.

## 📄 Licencia
[MIT](LICENSE)

---

**Nota Final:** Este proyecto es educativo. Para producción utiliza librerías profesionales como **scikit-learn**, **TensorFlow**, o **PyTorch**.
