from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

cali_data = fetch_california_housing()

X = cali_data.data
y = cali_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


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
            y_pred = X @ self.w + self.b
            error = y - y_pred
            loss = np.mean(error**2)

            # Aplico el descenso de gradiente
            dw = -(2/n_samples) * (X.T @ error)
            db = -(2/n_samples) * np.sum(error)

            # Aplico los cambios a mis parametros
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # Metodo de control
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}: Loss = {loss:.6f}')

    def predict(self, X):
        '''
        Funcion de prediccion de resultados
        '''
        return X @ self.w + self.b

    def evaluate_MSE(self, X, y):
        '''
        Funcion evaluacion de resultados con MSE
        '''
        y_pred = self.predict(X)
        return np.mean((y - y_pred**2))