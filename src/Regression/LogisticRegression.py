import numpy as np
from typing import Optional

'''
En este proyecto, implementare la regresion logistica basica multivariante utilizando la
libreria NumPy

Para la evaluacion implementare las metricas de:
    - Accuracy
    - Precision
    - F1 Score
    - Recall

Ademas de la matriz de confusion

'''


class LogisticRegression():
    def __init__(self, epoch=10000, lr=0.0001):
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.epoch: int = epoch
        self.lr = lr
    
    # Funcion sigmoidea 1/(1+e**-z)
    # donde z = W * X + b

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w: np.ndarray = np.zeros(n_features)
        self.b: float = 0.0

        for epoch in range(self.epoch):
            '''
            Por cada epoch:
             - Calculo la funcion de regresion logistica
             - Calculo la funcion sigmoide
             - Calculo la perdida (L)
             - Calculo los gradientes
             - Actualizo los parametros (W y b)
            '''

            z = X @ self.w + self.b
            y_pred = self.sigmoid(z)
            error = y_pred - y

            log_loss = -(1/n_samples) * np.sum(y * np.log(y_pred) + (1-y)*np.log(1-y_pred))

            # Descenso de gradiente
            dw = (1/n_samples) * X.T @ error
            db = (1/n_samples) * np.sum(error)

            # Actualizacion
            self.w -= self.lr * dw
            self.b -= self.lr * db 

            # Control
            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Log_Loss = {log_loss:.6f}')

    def predict(self, X) -> np.ndarray:
        z = X @ self.w + self.b
        y_pred = self.sigmoid(z)
        y_output= np.array(y_pred >= 0.5).astype(int)  # Mascara vectorizada para el output
        return y_output

    def predict_proba(self, X) -> np.ndarray:
        '''
        Devuelve la probabilidad de cada clase para cada muestra en X
        '''
        z = X @ self.w + self.b
        y_pred = self.sigmoid(z)
        return y_pred



# Datos de test


X_train = np.array([[10], [20], [90], [80], [30], [60], [70], [40]])
y_train = np.array([0, 0, 1, 1, 0, 1, 1, 0])

X_test = np.array([[25], [65], [55], [15]])
y_test = np.array([0, 1, 1, 0])


model = LogisticRegression(epoch=100000)

model.fit(X_train, y_train)

model.predict(X_test)