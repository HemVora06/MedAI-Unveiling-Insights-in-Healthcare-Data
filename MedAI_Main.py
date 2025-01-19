import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Importing the dataset
cancer_data=pd.read_csv('Breast_Cancer_Dataset.csv')

print(cancer_data.head())

#Understanding the data
print(cancer_data.info())
print(cancer_data.describe())
print(cancer_data.columns)
print(cancer_data.dtypes)

#Checking the Dataset for Null Values
print(cancer_data.isnull().sum())

#Dropping the unnamed Column
cancer_data.drop(columns='Unnamed: 32', inplace=True)

#Checking fot Null Values Again
print(cancer_data.isnull().sum())

#Target Variable and Features for Machine Learning
target = cancer_data['diagnosis']
features = cancer_data[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']]

#Converting Categorical Features into category DType
cancer_data['diagnosis']=cancer_data['diagnosis'].astype('category')
print(cancer_data.dtypes)

#Calculating the proportion of Benign Tumours to Malignant Tumours
proportions= cancer_data['diagnosis'].value_counts(normalize=True)
print(proportions)