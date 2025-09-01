import numpy as np
from typing import Optional


# Formula y_pred = X_test @ w + b
# w: weights
# b: bias

'''
En este caso, la funcion de coste tiene un anadido, tal que queda de la siguiente manera:

J(θ) = (1/2m) * Σ(y_pred - y)² + λ * R(θ)

Donde λ es el parametro llamado alpha que controla la regularizacion y R(θ) representa una funcion
de la suma de la suma del valor absoluto de todos los pesos del modelo(w) 
'''




class LassoRegression():
    def __init__(self, epochs: int=1000, lr: float=0.001, alpha: float=0.5) -> None:
        if epochs <= 0:
            raise ValueError(f'epochs debe ser un entero positivo, recibido: {epochs}')
        if lr <= 0:
            raise ValueError(f'learning_rate debe ser un número positivo, recibido: {lr}')
        
        self.epochs: int = epochs
        self.lr: float = lr
        self.weights: Optional[np.ndarray] | None = None  # el valor es None dado que no conozco las dimensiones de X_test
        self.bias: float = 0.0
        self.alpha: float = alpha
    
    def fit(self, X: np.ndarray, y: np.ndarray)-> None:
        X, y = self._validate_inputs(X=X, y=y, check_fitted=False)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for epoch in range(self.epochs):
            '''
            Por cada Epoch:
            - Defino mi funcion de la regresion lineal
            - Calculo la matriz de error.
            - Calculo el MSE.
            '''
            y_pred = X @ self.weights + self.bias
            error = y - y_pred
            self.J = np.sum(abs(self.weights))
            loss = np.mean(error**2) + self.alpha * self.J

            

            # Aplico el descenso de gradiente
            dw = -(2/n_samples) * (X.T @ error) + self.alpha * np.sign(self.weights)
            db = -(2/n_samples) * np.sum(error)

            # Aplico los cambios a mis parametros
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Metodo de control
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}: Loss = {loss:.6f}')

    def predict(self, X: np.ndarray)-> np.ndarray:
        '''
        Funcion de prediccion de resultados
        '''
        X, _ = self._validate_inputs(X=X)
        return X @ self.weights + self.bias

    def evaluate_mse(self, X: np.ndarray, y: np.ndarray)-> np.float64:
        '''
        Funcion evaluacion de resultados con MSE
        '''
        X, y = self._validate_inputs(X=X, y=y)
        y_pred = self.predict(X)
        return np.mean((y - y_pred)**2) + self.alpha * self.J

    def _validate_inputs(self, X, y=None, check_fitted=True):
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