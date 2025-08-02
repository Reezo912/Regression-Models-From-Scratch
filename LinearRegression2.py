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
class LinealRegresion():
    def __init__(self, epoch=1000, lr=0.001) -> None:
        self.lr = lr
        self.epoch = epoch
        self.w = None  # el valor es None dado que no conozco las dimensiones de X_test
        self.b: float = 0.0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b: float = 0.0

        for epoch in range(self.epoch):
            '''
            Por cada Epoch:
            - Defino mi funcion de la regresion lineal
            - Calculo la matriz de error.
            - Calculo el MSE.
            '''
            y_pred = self.w @ X + self.b
            error = y - y_pred
            loss = np.mean(error**2)

            # Aplico el descenso de gradiente
            dw = -(2/smaple)

