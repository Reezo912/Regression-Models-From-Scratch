# Registro de Cambios

Todos los cambios notables a este proyecto serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
y este proyecto adhiere a [Versionado Semántico](https://semver.org/spec/v2.0.0.html).

## [Sin Publicar]

### Planificado para v0.0.4
#### Añadido
- Mecanismo de parada temprana para convergencia de entrenamiento
- Programación de decaimiento de tasa de aprendizaje
- Parámetro de control de verbosidad para salida más limpia
- Seguimiento de métricas de convergencia
- PCA

#### Corregido
- Consistencia de nombres entre atributos de modelo y pruebas
- Mensajes de error mejorados para mejor depuración

#### Cambiado
- Eficiencia de entrenamiento mejorada con parada temprana

## [0.0.3] - 2025-01-04

### Añadido
- Script de ejecución profesional con salida formateada (`scripts/run_all.py`)
- Formateo automático de decimales para visualización limpia de métricas
- Estructura de proyecto comprehensiva con directorio de scripts
- Validación robusta de entrada para ambos modelos de regresión

### Corregido
- Error crítico en validación de entrada de LogisticRegression
- Comparación incorrecta con None en validación de modelo
- Configuración de ruta en gitignore
- Eliminado import no utilizado de multiprocessing

### Cambiado
- Reescritura completa del README con tablas de resultados cuantificados
- Documentación mejorada con métricas de rendimiento reales
- Organización mejorada de estructura de proyecto

### Rendimiento
- Regresión Lineal: 99.98% precisión de sklearn (MSE: 0.5560 vs 0.5559)
- Regresión Logística: 97.90% precisión en dataset Breast Cancer

## [0.0.2] - 2025-01-04

### Añadido
- Implementaciones básicas de Regresión Lineal y Logística
- Pruebas comprehensivas con comparación contra scikit-learn
- Licencia MIT y documentación apropiada

### Corregido
- Errores de implementación inicial

## [0.0.1] - 2025-01-03

### Añadido
- Inicialización del proyecto
- Estructura básica del proyecto
- Borradores de implementación inicial