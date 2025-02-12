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
        self.corv_matrix = None
        self.eigenvector = None
        self.eigenvalue = None
        self.total_variance = None
        self.explained_variance_ratio = None
        self.cumuative_explained_variance = None
        self._selected_components = None
        self.transformed_data = None

    def standardizing_data(self, features = None):
        if features == None:
            features = self.data.select_dtypes(include=[np.number]).columns
            self.standardized_data = (self.data[features] - self.data[features].mean()) / self.data[features].std()
        else:
            self.standardized_data = (self.data[features] - self.data[features].mean()) / self.data[features].std()
        return self.standardized_data
    def compute_covariance(self):
        self.corv_matrix = np.cov(self.standardized_data, rowvar = False)
        return self.corv_matrix
    def compute_eigen(self):
        self.eigenvalue, self.eigenvector = np.linalg.eig(self.corv_matrix)
        return self.eigenvalue, self.eigenvector
    def sort_eigen(self):
        sorted_indices = np.argsort(self.eigenvalue)[::-1]
        self.eigenvalue = self.eigenvalue[sorted_indices]
        self.eigenvector = self.eigenvector[:, sorted_indices]
        return self.eigenvalue, self.eigenvector
    def select_components(self):
        self.total_variance = sum(self.eigenvalue)
        self.explained_variance_ratio = self.eigenvalue / self.total_variance
        self.cumuative_explained_variance = np.cumsum(self.explained_variance_ratio)
        self._selected_components = self.eigenvector[:, : self.n_components]
        return self._selected_components
    def transform(self, features = None):
        if features == None:
            features = self.data.select_dtypes(include=[np.number]).columns        
        centered_data = self.data[features] - self.data[features].mean()
        self.transformed_data = centered_data.dot(self._selected_components)
        return self.transformed_data
    def fit_transform(self, features = None):
        if features == None:
            features = self.data.columns
        self.standardizing_data(features = None)
        self.compute_covariance()
        self.compute_eigen()
        self.sort_eigen()
        self.select_components()
        return self.transform(features = None)
    
#Making Train Test Split
class train_test_split():
    def __init__(self, test_size = 0.25, train_size = None, shuffle = True, random_state = None):
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.train_size = train_size
        self._n_samples = None
        self.data = None
        self.target = None

    def data_validater(self, data, target):
        #Check if the length of data and target matches
        if len(data) != len(target):
            raise ValueError("Data and Target must have the same number of rows")
        #Check if the format of data is valid
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, (list, np.ndarray)):
            self.data = pd.DataFrame(data)
        else:
            raise ValueError("Data must be either a Dataframe or a List or a Ndarray")
        #Check if the format of target is correct
        if isinstance(target, (pd.Series, np.ndarray, list)):
            self.target = pd.Series(target) if not isinstance(target, pd.Series) else target
        else:
            raise ValueError("Data must be either Series, Ndarray or List")
            
    def split(self, data, target):
        self.data_validater(data, target)
        self._n_samples = len(data)
        self.indices = np.arange(self._n_samples)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if type(self.test_size) is float and self.train_size == None:
            self._n_test = int(self.test_size * self._n_samples)
            self._n_train = self._n_samples - self._n_test
        elif type(self.test_size) is int and self.train_size == None:
            self._n_test = self.test_size
            self._n_train = self._n_samples - self._n_test
        elif self.test_size == None and self.train_size == None:
            self._n_test = int(self.test_size * self._n_samples)
            self._n_train = self._n_samples - self._n_test
        elif self.test_size == None and type(self.train_size) is float:
            self._n_train = int(self.train_size * self._n_samples)
            self._n_test = self._n_samples - self._n_train
        elif self.test_size == None and type(self.train_size) is int:
            self._n_train = self.train_size
            self._n_test = self._n_samples - self._n_train
        elif type(self.test_size) is float and type(self.train_size) is float:
            self._n_test = int(self.test_size * self._n_samples)
            self._n_train = int(self.train_size * self._n_samples)
        elif type(self.test_size) is int and type(self.train_size) is int:
            self._n_test = self.test_size
            self._n_train = self.train_size
        elif type(self.test_size) is int and type(self.train_size) is float:
            self._n_train = int(self.train_size * self._n_samples)
            self._n_test = self.test_size
        elif type(self.test_size) is float and type(self.train_size) is int:
            self._n_train = self.train_size
            self._n_test = int(self.test_size * self._n_samples)
        else:
            raise ValueError("HOW???")
        if self._n_test + self._n_train > self._n_samples:
            raise ValueError("Test size and train size cannot be larger than the total number of samples")
        if self.shuffle == True:
            self.shuffled_indices = np.random.permutation(self.indices)
            self.test_indices = self.shuffled_indices[:self._n_test]
            self.train_indices = self.shuffled_indices[self._n_test:]
        else:
            self.test_indices = self.indices[:self._n_test]
            self.train_indices = self.indices[self._n_test:]
        self.X_train = data.iloc[self.train_indices]
        self.y_train = target.iloc[self.train_indices]
        self.X_test = data.iloc[self.test_indices]
        self.y_test = target.iloc[self.test_indices]
        return self.X_train, self.X_test, self.y_train, self.y_test