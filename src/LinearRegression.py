import numpy as np


# Formula y_pred = X_test @ w + b
# w: pesos
# b: bias

'''
En este proyecto, implementare la regresion lineal basica multivariante utilizando la
libreria NumPy

 Para esto utilizaré la funcion de descenso de gradiente para minimizar la funcion de error,
 que en este caso será MSE(Error cuadrático medio)

 '''

class LinearRegression():
    def __init__(self, epoch: int=1000, lr: float=0.001) -> None:
        if epoch <= 0:
            raise ValueError('epoch debe ser mayor que 0')
        if lr <= 0:
            raise ValueError('lr debe ser mayor que 0')

        self.lr: float = lr
        self.epoch: int = epoch
        self.pesos = None  # el valor es None dado que no conozco las dimensiones de X_test
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray)-> None:
        
        X, y = self._validate_inputs(X=X, y=y, check_fitted=False)
        n_samples, n_features = X.shape
        self.pesos = np.zeros(n_features)
        self.bias: float = 0.0

        for epoch in range(self.epoch):
            '''
            Por cada Epoch:
            - Defino mi funcion de la regresion lineal
            - Calculo la matriz de error.
            - Calculo el MSE.
            '''
            y_pred = X @ self.pesos + self.bias
            error = y - y_pred
            loss = np.mean(error**2)

            # Aplico el descenso de gradiente
            dw = -(2/n_samples) * (X.T @ error)
            db = -(2/n_samples) * np.sum(error)

            # Aplico los cambios a mis parametros
            self.pesos -= self.lr * dw
            self.bias -= self.lr * db

            # Metodo de control
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}: Loss = {loss:.6f}')

    def predict(self, X: np.ndarray)-> np.ndarray:
        '''
        Funcion de prediccion de resultados
        '''
        X, _ = self._validate_inputs(X=X)
        return X @ self.pesos + self.bias

    def evaluate_mse(self, X: np.ndarray, y: np.ndarray)-> np.float64:
        '''
        Funcion evaluacion de resultados con MSE
        '''
        X, y = self._validate_inputs(X=X, y=y)
        y_pred = self.predict(X)
        return np.mean((y - y_pred)**2)

    def _validate_inputs(self, X, y=None, check_fitted=True):
        '''
        Funcion para validar los inputs y evitar fallos ademas de dar feedback para correccion.
        '''
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if y is not None and not isinstance(y, np.ndarray):
            y = np.array(y)
        
        if X.size == 0:
            raise ValueError('X no puede estar vacío')
        if y is not None and y.size == 0:
            raise ValueError('y no puede estar vacío')

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