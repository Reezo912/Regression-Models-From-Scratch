import numpy as np
from typing import Optional

'''
En este proyecto, implementare la regresion logistica binaria basica multivariante utilizando la
libreria NumPy

En un futuro mi idea es implementar un sistema multiclase utilizando One-vs-Rest (OvR) y softmax regression

Para la evaluacion implementare las metricas de:
    - Accuracy
    - Precision
    - F1 Score
    - Recall

Ademas de la matriz de confusion

'''


class LogisticRegression():
    def __init__(self, epochs: int=10000, lr: float=0.0001)-> None:
        if epochs <= 0:
            raise ValueError(f'epochs debe ser un entero positivo, recibido: {epochs}')
        if lr <= 0:
            raise ValueError(f'learning_rate debe ser un número positivo, recibido: {lr}')

        self.epochs: int = epochs
        self.lr: float = lr
        self.weights: Optional[np.ndarray] | None = None
        self.bias: float = 0.0

    
    # Funcion sigmoidea 1/(1+e**-z)
    # donde z = W * X + b

    def sigmoid(self, z: np.ndarray)-> np.ndarray:
        return 1/(1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray | None) -> None:

        X, y = self._validate_inputs(X=X, y=y, check_fitted=False)
        n_samples, n_features = X.shape
        self.weights: np.ndarray = np.zeros(n_features)
        self.bias: float = 0.0

        for epoch in range(self.epochs):
            '''
            Por cada epoch:
             - Calculo la funcion de regresion logistica
             - Calculo la funcion sigmoide
             - Calculo la perdida (L)
             - Calculo los gradientes
             - Actualizo los parametros (W y b)
            '''

            epsilon = 1e-15
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)
            # Para evitar que por redondeo y_pred=0 añado un clipping
            y_pred = np.clip(y_pred, epsilon, 1-epsilon)
            error = y_pred - y

            log_loss = -(1/n_samples) * np.sum(y * np.log(y_pred) + (1-y)*np.log(1-y_pred))

            # Descenso de gradiente
            dw = (1/n_samples) * X.T @ error
            db = (1/n_samples) * np.sum(error)

            # Actualizacion
            self.weights -= self.lr * dw
            self.bias -= self.lr * db 

            # Control
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}: Log_Loss = {log_loss:.6f}')

    def predict(self, X: np.ndarray) -> np.ndarray:
        X, _ = self._validate_inputs(X=X)
        z = X @ self.weights + self.bias
        y_pred = self.sigmoid(z)
        y_output= np.array(y_pred >= 0.5).astype(int)  # Mascara vectorizada para el output
        return y_output

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        '''
        Devuelve la probabilidad de cada clase para cada muestra en X
        '''
        X, _ = self._validate_inputs(X=X)
        z = X @ self.weights + self.bias
        y_pred = self.sigmoid(z)
        return y_pred

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray | None = None, check_fitted=True):
        '''
        Funcion para validar los inputs y evitar fallos ademas de dar feedback para correccion.
        '''
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if y is not None and not isinstance(y, np.ndarray):
            y = np.array(y)
        
        if X.size == 0:
            raise ValueError('X no puede estar vacío. Proporciona una matriz con al menos una muestra.')
        if y is not None and y.size == 0:
            raise ValueError('y no puede estar vacío. Proporciona un vector con al menos una etiqueta.')
        
        if y is not None:
            if y.ndim != 1:
                raise ValueError(f'y debe ser un vector 1D (shape (n,)), pero recibió forma {y.shape}. '
                               f'Usa y.ravel() o y.reshape(-1) para aplanar.')
            if X.shape[0] != y.shape[0]:
                raise ValueError(f'X e y deben tener el mismo número de muestras. '
                               f'X: {X.shape[0]} muestras, y: {y.shape[0]} muestras.')

        if check_fitted and self.weights is None:
            raise ValueError('El modelo no ha sido entrenado. Llama primero al método fit(X, y) '
                           'con tus datos de entrenamiento.')
        
        if check_fitted and self.weights is not None:
            if X.shape[1] != self.weights.shape[0]:
                raise ValueError(f'Inconsistencia en número de características. '
                               f'Modelo entrenado con {self.weights.shape[0]} características, '
                               f'pero datos de entrada tienen {X.shape[1]}. '
                               f'Verifica que uses el mismo preprocesamiento.')

        return X, y