import pandas as pd
import numpy as np

#Standard Scaler
class StandardScaler():
    def __init__(self):
        self._mean = None
        self._std_dev = None
        self._standardized = None

    def fit(self, data, features):
        self._mean = data[features].mean()
        self._std_dev = data[features].std()
        return self._mean, self._std_dev

    def transform(self, data, features):
        self._standardized = (data[features] - self._mean) / self._std_dev
        return self._standardized
    
    def fit_transform(self, data, features):
        self.fit(data, features)
        return self.transform(data, features)
    
#Min-Max Scaler
class MinMaxScaler():
    def __init__(self, feature_range):
        self.feature_range = feature_range
        self._min = None
        self._max = None
    def fit(self, data, features):
        self._min = data[features].min()
        self._max = data[features].max()
        return self._min, self._max
    def transform(self, data, features):
        scaled = (data[features] - self._min) / (self._max - self._min)
        normalized = scaled * (self.feature_range[0] + self.feature_range[1]) + self.feature_range[0]
        return normalized
    def fit_transform(self, data, features):
        self.fit(data, features)
        return self.transform(data, features)
    
#Principal Component Analysis
class PCA():
    def __init__(self, n_components, data):
        self.n_components = n_components
        self.data = data
        self.standardized_data = None
        self._corv_matrix = None
        self._eigenvector = None
        self._eigenvalue = None
        self.total_variance = None
        self._explained_variance_ratio = None
        self._cumuative_explained_variance = None
        self._selected_components = None
        self.transformed_data = None

    def standardizing_data(self, data, features = None):
        if features == None:
            features = data.select_dtypes(include=[np.number]).columns
            self.standardized_data = (data[features] - data[features].mean()) / data[features].std()
        else:
            self.standardized_data = (data[features] - data[features].mean()) / data[features].std()
        return self.standardized_data
    def compute_covariance(self):
        self._corv_matrix = np.cov(self.standardized_data, rowvar = False)
        return self._corv_matrix
    def compute_eigen(self):
        self._eigenvalue, self._eigenvector = np.linalg.eig(self._corv_matrix)
        return self._eigenvalue, self._eigenvector
    def sort_eigen(self):
        sorted_indices = np.argsort(self._eigenvalue)[::-1]
        self._eigenvalue = self._eigenvalue[sorted_indices]
        self._eigenvector = self._eigenvector[:, sorted_indices]
        return self._eigenvalue, self._eigenvector
    def select_components(self):
        self.total_variance = sum(self._eigenvalue)
        self._explained_variance_ratio = self._eigenvalue / self.total_variance
        self._cumuative_explained_variance = np.cumsum(self._explained_variance_ratio)
        self._selected_components = self._eigenvector[:, : self.n_components]
        return self._selected_components
    def transform(self, data, features = None):
        if features == None:
            features = data.select_dtypes(include=[np.number]).columns        
        centered_data = data[features] - data[features].mean()
        self.transformed_data = centered_data.dot(self._selected_components)
        return self.transformed_data
    def fit_transform(self, data, features = None):
        if features == None:
            features = data.columns
        self.standardizing_data(data, features = None)
        self.compute_covariance()
        self.compute_eigen()
        self.sort_eigen()
        self.select_components()
        return self.transform(data, features = None)