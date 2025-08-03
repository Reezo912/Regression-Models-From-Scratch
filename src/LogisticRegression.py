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
    def __init__(self, epoch: int=10000, lr: float=0.0001)-> None:
        if epoch <= 0:
            raise ValueError('el valor de epoch debe ser mayor que 0')
        if lr <= 0:
            raise ValueError('el valor de lr debe ser mayor que 0')

        self.epoch: int = epoch
        self.lr: float = lr
        self.pesos: Optional[np.ndarray] = None
        self.bias: float = 0.0

    
    # Funcion sigmoidea 1/(1+e**-z)
    # donde z = W * X + b

    def sigmoid(self, z: np.ndarray)-> np.ndarray:
        return 1/(1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

        X, y = self._validate_inputs(X=X, y=y, check_fitted=False)
        n_samples, n_features = X.shape
        self.pesos: np.ndarray = np.zeros(n_features)
        self.bias: float = 0.0

        for epoch in range(self.epoch):
            '''
            Por cada epoch:
             - Calculo la funcion de regresion logistica
             - Calculo la funcion sigmoide
             - Calculo la perdida (L)
             - Calculo los gradientes
             - Actualizo los parametros (W y b)
            '''

            epsilon = 1e-15
            z = X @ self.pesos + self.bias
            y_pred = self.sigmoid(z)
            # Para evitar que por redondeo y_pred=0 añado un clipping
            y_pred = np.clip(y_pred, epsilon, 1-epsilon)
            error = y_pred - y

            log_loss = -(1/n_samples) * np.sum(y * np.log(y_pred) + (1-y)*np.log(1-y_pred))

            # Descenso de gradiente
            dw = (1/n_samples) * X.T @ error
            db = (1/n_samples) * np.sum(error)

            # Actualizacion
            self.pesos -= self.lr * dw
            self.bias -= self.lr * db 

            # Control
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}: Log_Loss = {log_loss:.6f}')

    def predict(self, X: np.ndarray) -> np.ndarray:
        X, _ = self._validate_inputs(X=X)
        z = X @ self.pesos + self.bias
        y_pred = self.sigmoid(z)
        y_output= np.array(y_pred >= 0.5).astype(int)  # Mascara vectorizada para el output
        return y_output

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        '''
        Devuelve la probabilidad de cada clase para cada muestra en X
        '''
        X, _ = self._validate_inputs(X=X)
        z = X @ self.pesos + self.bias
        y_pred = self.sigmoid(z)
        return y_pred

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray = None, check_fitted=True):
        '''
        Funcion para validar los inputs y evitar fallos ademas de dar feedback para correccion.
        '''
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if y is not None and not isinstance(y, np.ndarray):
            y = np.array(y)
        
        if X.size == 0:
            raise ValueError('X no puede estar vacio')
        if y is not None and y.size == 0:
            raise ValueError('y no puede estar vacio')
        
        if y is not None:
            if y.ndim != 1:
                raise ValueError('y debe ser un vector 1D')
            if X.shape[0] != y.shape[0]:
                raise ValueError('X e y deben tener el mismo número de muestras')

        if check_fitted and self.pesos is None:
            raise ValueError('El modelo no ha sido entrenado')
        
        if check_fitted and self.pesos is not None:
            if X.shape[1] != self.pesos.shape[0]:
                raise ValueError(f'Esperadas {self.pesos.shape[0]} features, obtenidas {X.shape[1]}')

        return X, y