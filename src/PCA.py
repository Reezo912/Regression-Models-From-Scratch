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
    def __init__(self, n_components) -> None:
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.mean_ = None
        self.std_ = None


    def _validate_inputs(self, X: np.ndarray)-> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if X.size == 0:
            raise ValueError('Los datos de entrada no pueden estar vacios.')

        if X.ndim == 1 or X.shape[1] < 2:
            raise ValueError('PCA solo funciona con arrays multivariantes, el array introducido tiene una sola dimension')
        
        return X
    
    def _standardize(self, X: np.ndarray)-> None:
        '''
        Devuelve la media, la std y el array standarizado
        '''
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        if np.any(std == 0):
            raise ValueError('Al menos una columna tiene desviación estándar igual a cero, no se puede estandarizar para PCA.')
        
        X_std = (X - mean) / std
        return X_std, mean, std

    def fit(self, X: np.ndarray)-> np.ndarray:
        '''
        Metodo para aplicar fit sobre la matriz dada, los resultados se guardan en el objeto
        '''
        X = self._validate_inputs(X)
        X_std, mean, std = self._standardize(X)
        self.mean_ = mean
        self.std_ = std

        # Matriz de covarianza
        n_rows = X.shape[0]
        cov_matrix = (X_std.T @ X_std) / (n_rows - 1)  # Para implementacion normal usar np.cov()
        # Extraigo autovectores y autovalores
        eig_val, eig_vec = np.linalg.eigh(cov_matrix)
        
        # Obtengo los indices ordenados de mayor a menor de los autovectores, para luego aplicarlos sobre los autovalores.
        idx = np.argsort(eig_val)[::-1]

        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]

        # Reduzco el numero de componentes a los solicitados con slice
        eig_val = eig_val[:self.n_components]
        eig_vec = eig_vec[:, :self.n_components]

        # Obtengo los componentes, la varianza y la media
        self.components_ = eig_vec
        self.explained_variance_ = eig_val

        return self

    
    def transform(self, X:np.ndarray)-> np.ndarray:
        '''
        Metodo para aplicar el transform, el objeto debe de haber hecho fit antes.
        Da como resultado un array
        '''
        if self.mean_ == None:
            raise Exception('Primero tienes que usar .fit() en el array')

        X = self._validate_inputs(X)
        X_std = (X - self.mean_) / self.std_
        return X_std @ self.components_
    
    def fit_transform(self, X: np.ndarray)-> np.ndarray:
        '''
        Metodo para aplicar el fit y el transform automaticamente
        '''

        self.fit(X)
        return self.transform(X)