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



# -----------------------
# Linear regression model
# -----------------------

# y_pred = X_test @ w + b 
# w : pesos
# b : bias

class LinearRegression():
    def __init__(self, lr=0.001, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None   # w = pesos
        self.b = 0.0    # b = bias

    # funcion de entrenamiento del modelo
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.epochs):
            y_pred = X @ self.w + self.b   # funcion de la recta de la regresion lineal
            error = y - y_pred             # Calculo mi error
            loss = np.mean(error**2)       # Funcion de perdida  MSE

            # gradient descent
            dw = -(2/n_samples) * (np.transpose(X) @ error)  # derivada de peso
            db = -(2/n_samples) * np.sum(error)              # derivada de bias

            # Actualizacion
            self.w -= self.lr *dw   # Se actualiza el peso despues de aplicar la correccion de gradient descent
            self.b -= self.lr *db   # Se actualiza el bias despues de aplicar la correccion de gradient descent

            # imprimo la epoch en la que estoy y mi valor de loss
            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Loss = {loss:.6f}')
    
    def predict(self, X):
        return X @ self.w + self.b    # Se hace una prediccion una vez los pesos y el bias han sido entrenados
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y-y_pred)**2)   # MSE
    


########

# Entrenamiento


model = LinearRegression(lr=0.0005, epochs=100000)

model.fit(X_train, y_train)

mse_train = model.evaluate(X_train, y_train)
mse_test = model.evaluate(X_test, y_test)

print(f'\nMSE (train): {mse_train:.6f}')
print(f'MSE (test): {mse_test:.6f}')

y_pred = model.predict(X_test)
print('prediccion ejemplo:', y_pred[:5])
print('Valor real:', y_test[:5])
print('Model peso:', model.w)
print('model bias:', model.b)



from sklearn.linear_model import LinearRegression
sk_model = LinearRegression()
sk_model.fit(X_train, y_train)

print("\nsklearn w:", sk_model.coef_)
print("sklearn b:", sk_model.intercept_)
print("MSE test sklearn:", np.mean((y_test - sk_model.predict(X_test))**2))
