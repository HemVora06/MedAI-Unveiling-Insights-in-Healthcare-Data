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

#Label Encoding Diagnosis Column for easier time
#Mapping the categorical values
diagnosis_map = {'B': 0, 'M': 1}

#Applying the map to the 'diagnosis' column
cancer_data['diagnosis'] = cancer_data['diagnosis'].map(diagnosis_map)

#Target, Features and Numeric Values for Machine Learning
target = cancer_data['diagnosis']
features = cancer_data[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']]
numeric_val=cancer_data.drop(columns=['diagnosis', 'id'], axis=1)

#Converting Categorical Features into category DType
cancer_data['diagnosis']=cancer_data['diagnosis'].astype('category')
print(cancer_data.dtypes)

#Calculating the proportion of Benign Tumours to Malignant Tumours
proportions= cancer_data['diagnosis'].value_counts(normalize=True)
print(proportions)

#Describing the numerical Data to find Outliers
print(cancer_data.describe(include=[np.number]))

#Finding Duplicate Rows
print(cancer_data.duplicated().sum())#No Duplicates

#Calculating the range, variance, and standard deviation for features

#Calculating Range
#Range is the Difference between the maximum and minimum values of a column
cancer_data_range=features.max(axis=0)-features.min(axis=0)
print('Range: ')
print(cancer_data_range)

#Calculating Variance
#Variance is the average of the sum of squared differences from the mean
#Variance is a measure of the spread of a set of data
cancer_data_variance=features.var(axis=0, numeric_only=True)
print('Variance')
print(cancer_data_variance)

#Calculating Standard Deviation
#Standard Deviation is the square root of the variance
#Standard Deviation is a measure of the amount of variation or dispersion of a set of values
cancer_data_std_dev=features.std(axis=0, numeric_only=True)
print('Standard Deviation')
print(cancer_data_std_dev)

#Investigating correlations between numerical features and the target variable
#Correlation is a measure of the relationship between two variables
corr_matrix = numeric_val.corrwith(target)
print('Correlation Matrix: \n', corr_matrix)
'''According to the corrlation matrix, these are the features that are either strongly or moderately positively related to target along with their values:
radius_mean                0.730029
perimeter_mean             0.742636
area_mean                  0.708984
compactness_mean           0.596534
concavity_mean             0.696360
concave points_mean        0.776614
radius_se                  0.567134
perimeter_se               0.556141
area_se                    0.548236
radius_worst               0.776454
perimeter_worst            0.782914
area_worst                 0.733825
compactness_worst          0.590998
concavity_worst            0.659610
concave points_worst       0.793566
The features will for now be replaced with these variables 
but in future when it is time for making ML models like Linear regression and/or 
at the time of Data preprocessing, redundant variables like 'primeter_mean', 'perimeter_worst', etc. 
will be combine or removed because these variables increase multicolinearity.
'''
features=[cancer_data[['radius_mean', 'perimeter_mean', 'area_mean', 'compactness_mean', 'concavity_mean', 
'concave points_mean', 'radius_se', 'perimeter_se', 'area_se', 'radius_worst', 'perimeter_worst', 
'area_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst']]]

#Creating a heatmap of correlations among all numerical features
corr_matrix = numeric_val.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

