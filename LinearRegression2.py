import numpy as np

# Formula y_pred = X_test @ w + b
# w: pesos
# b: bias

''' Conociendo esto, implementare la regresion lineal basica multivariante ytilizando la
 libreria NumPy'''

class LinealRegresion():
    def __init__(self, epoch=1000, lr=0.001) -> None:
        self.lr = lr
        self.epoch = epoch
        self.w = None  # el valor es None dado que no conozco las dimensiones de X_test
        self.b: float = 0.0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        