
# Mi Propia Regresión Lineal y Logística

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-green)](https://www.python.org/downloads/)
[![NumPy Version](https://img.shields.io/badge/NumPy-1.26%2B-yellow)](https://numpy.org/)

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
├── tests/
│   ├── test_linear.py           # Pruebas para regresión lineal
│   └── test_logistic.py         # Pruebas para regresión logística
└── scripts/
    └── run_all.py               # Script para ejecutar todos los tests
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

### Ejecución Completa (Recomendada)
```bash
# Ejecuta todos los tests con formato profesional
python scripts/run_all.py
```

### Ejecución Individual
```bash
# Test específico de regresión lineal
python tests/test_linear.py

# Test específico de regresión logística  
python tests/test_logistic.py
```

## 📊 Resultados Obtenidos

### Regresión Lineal (California Housing)
| Métrica | scikit-learn | Mi Implementación | Diferencia |
|---------|--------------|-------------------|------------|
| **MSE (train)** | 0.5179 | 0.5179 | 0.0000 |
| **MSE (test)** | 0.5559 | 0.5560 | +0.0001 |

✅ **Rendimiento prácticamente idéntico** - Diferencia de solo 0.02% en test set

### Regresión Logística (Breast Cancer Wisconsin)  
| Métrica | scikit-learn | Mi Implementación | Diferencia |
|---------|--------------|-------------------|------------|
| **Accuracy** | 98.60% | 97.90% | -0.70% |
| **Precision** | 98.89% | 98.88% | -0.01% |
| **Recall** | 98.89% | 97.78% | -1.11% |
| **F1-Score** | 98.89% | 98.32% | -0.57% |

✅ **Excelente rendimiento** - Solo 0.7% de diferencia en accuracy con implementación desde cero

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

### 🔥 v0.0.4 - Mejoras Críticas (En progreso)
**Objetivo**: Optimizar implementaciones actuales para máxima robustez

**Prioridad Alta:**
- [ ] 🐛 Arreglar consistencia de nombres (`weights` vs `pesos`)
- [ ] ⚡ Implementar parada temprana (early stopping)
- [ ] 📈 Programación de decaimiento de tasa de aprendizaje
- [ ] 🔧 Parámetro de control de verbosidad
- [ ] 📊 Seguimiento de métricas de convergencia
- [ ] ⚠️ Mejores mensajes de error

**Resultado esperado**: Implementaciones perfectas y profesionales

### 🚀 v0.1.0 - Nuevos Algoritmos (Planificado)
**Objetivo**: Expandir familia de algoritmos de regresión

**Nuevas Funcionalidades:**
- [ ] 🎯 Regresión Ridge (regularización L2)
- [ ] 🎯 Regresión Lasso (regularización L1)  
- [ ] 🎯 ElasticNet (combinación L1 + L2)
- [ ] 📐 Soporte para características polinómicas
- [ ] 🔄 Marco de validación cruzada

### 💡 v0.2.0 - Algoritmos Avanzados (Futuro)
**Objetivo**: Machine Learning más allá de regresión

**Algoritmos Planificados:**
- [ ] 🤖 Máquinas de Vectores de Soporte (SVM)
- [ ] 🌳 Árboles de Decisión desde cero
- [ ] 🔍 Agrupamiento K-Means
- [ ] 📊 Análisis de Componentes Principales (PCA)
- [ ] 🧠 Red Neuronal (Perceptrón)

### 🛠️ Mejoras de Infraestructura (Continuas)
- [ ] 📈 Suite de evaluación de rendimiento
- [ ] 🎨 Herramientas de visualización de datos
- [ ] 📦 Pruebas automatizadas con GitHub Actions
- [ ] 📚 Documentación de API con Sphinx
- [ ] 🐳 Contenedorización con Docker

## 🤝 Cómo Contribuir
¡Tu contribución es bienvenida!
- Abre un Issue con mejoras.
- Envía un Pull Request.
- Mejora documentación y agrega ejemplos.

## 📄 Licencia
[MIT](LICENSE)

---

**Nota Final:** Este proyecto es educativo. Para producción utiliza librerías profesionales como **scikit-learn**, **TensorFlow**, o **PyTorch**.
