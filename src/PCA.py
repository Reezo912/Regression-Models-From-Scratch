import numpy as np


'''
Mi implementacion del PCA utilizando numpy.

Esta implementacion la utilizare mas adelante en mi proyecto de Churn con KKBox para comprobar rendimiento del modelo.
'''

'''
PCA es muy sensible a diferentes escalas, por lo tanto lo primero sera asegurarme de que mis datos estan estandarizados

X_std = (X - media) / std_dev
'''



class PCA():
    def __init__(self) -> None:
        pass

    def _validate_inputs(self, X: np.ndarray)-> None:
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if X.size == 0:
            raise ValueError('Los datos de entrada no pueden estar vacios.')

        if X.ndim == 1 or X.shape[1] < 2:
            raise ValueError('PCA solo funciona con arrays multivariantes, el array introducido tiene una sola dimension')
        
        return X
    
    def _standardize(self, X: np.ndarray)-> np.ndarray:
        if np.any(X.std(axis=0) == 0):
            raise ValueError('Al menos una columna tiene desviación estándar cero, no se puede estandarizar para PCA.')
        
        X_std = (X - X.mean(axis=0)) / X.std(axis=0)
        return X_std

    def fit(self, X: np.ndarray)-> np.ndarray:
        X = self._validate_inputs(X)
        X_std = self._standardize(X)

        # Matriz de covarianza
        n_rows = X.shape[0]
        cov_matrix = (X_std.T @ X_std) / (n_rows - 1)
        eig_val, eig_vec = np.linalg.eig(cov_matrix)
        print(eig_val, eig_vec)





# ------------------------

# Test


X = [[0, 3 ,2], [2, 1, 3]]


pca = PCA()

pca.fit(X)