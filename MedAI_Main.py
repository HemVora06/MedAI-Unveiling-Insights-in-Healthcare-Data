import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import MedAI_functions as func

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
numeric_features=cancer_data.drop(columns=['diagnosis', 'id'], axis=1)

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
corr_matrix = numeric_features.corrwith(target)
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
corr_matrix = numeric_features.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

#Exploratory Data Analysis and Basic Data Exploration

#Plotting Histograms for Radius Mean

plt.figure(figsize=(10, 6))
sns.histplot(cancer_data['radius_mean'], kde=True)
plt.title('Histogram of Radius Mean')

#Plotting Histograms for Texture Mean

plt.figure(figsize=(10, 6))
sns.histplot(cancer_data['texture_mean'], kde=True)
plt.title('Histogram of Texture Mean')

#Plotting Histogram for Area Mean
plt.figure(figsize=(10, 6))
sns.histplot(cancer_data['area_mean'], kde=True)
plt.title('Histogram of Area Mean')

#Visualizing the Relationship between 'Radius Mean' and 'Area Mean' based on the Diagnosis on a Scatter Pot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='radius_mean', y='area_mean', data=cancer_data, hue='diagnosis')
plt.title('Scatter Plot of Radius Mean vs Area Mean based on Diagnosis')

#Grouping data by the "diagnosis" column and computing the mean for all other features.
grouped_data = cancer_data.groupby('diagnosis', observed=True).mean()
print(grouped_data)

#Using a box plot to compare `perimeter_mean` between benign and malignant cases.
plt.figure(figsize=(10, 6))
sns.boxplot(x='diagnosis', y='perimeter_mean', data=cancer_data)
plt.title('Box Plot of Perimeter Mean based on Diagnosis')
plt.xlabel('0-Benign, 1-Malignant', fontsize=16)

#Calculating the Mode of the Diagnosis
normal_diagnosis=cancer_data['diagnosis'].astype('int')
print('Mode of Diagnosis(0-Benign, 1-Malignant): ', normal_diagnosis.mode()[0])

#Creating a violin plot to show the distribution of `radius_mean` for each `diagnosis` type
plt.figure(figsize=(10, 6))
sns.violinplot(x='diagnosis', y='radius_mean', data=cancer_data)
plt.title('Violin Plot of Radius Mean based on Diagnosis')

#Grouping data by Diagnosis and plotting average 'Radius Mean' in a Bar Plot
grouped_data = cancer_data.groupby('diagnosis', observed=True)['radius_mean'].mean()
plt.figure(figsize=(10, 6))
sns.barplot(x=grouped_data.index, y=grouped_data.values)
plt.title('Average Grouped Radius Mean')

#Exploring the distribution of `texture_mean` using a KDE plot.
plt.figure(figsize=(10, 6))
sns.kdeplot(cancer_data['texture_mean'], shade=True)
plt.title('KDE Plot of Texture Mean')

#Plotting Diagnosis column in a Bar Plot
plt.figure(figsize=(10, 6))
sns.countplot(x='diagnosis', data=cancer_data)
plt.title('Bar Plot of Diagnosis')

#Creating a Pair plot of 'Radius Mean', 'Texture Mean' and 'Area Mean' with the hue of 'Diagnosis'
plt.figure(figsize=(10, 6))
sns.pairplot(data=cancer_data, vars=['radius_mean', 'texture_mean', 'area_mean'], hue='diagnosis')

#Grouping data by Diagnosis and plotting average 'Compactness Mean' in a Bar Plot
grouped_data = cancer_data.groupby('diagnosis', observed=True)['compactness_mean'].mean()
plt.figure(figsize=(10, 6))
sns.barplot(x=grouped_data.index, y=grouped_data.values)
plt.title('Average Grouped Compactness Mean')

#Creating a pivot table to show the average `area_mean` and `perimeter_mean` for each `diagnosis`
pivot_table = pd.pivot_table(cancer_data, values=['area_mean', 'perimeter_mean'], index='diagnosis', aggfunc='mean')
print(pivot_table)

#Creating a violin plot to show the distribution of `smoothness_mean` for each `diagnosis` type
plt.figure(figsize=(10, 6))
sns.violinplot(x='diagnosis', y='smoothness_mean', data=cancer_data)
plt.title('Violin Plot of Smoothness Mean based on Diagnosis')

#Visualizing the Relationship between 'Radius Mean' and 'Texture Mean' based on the Diagnosis on a Scatter Pot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='radius_mean', y='texture_mean', data=cancer_data, hue='diagnosis')
plt.title('Scatter Plot of Radius Mean vs Texture Mean based on Diagnosis')

# Group by `diagnosis` and visualize the average value of top 5 features with highest correlation to the target
corr_matrix=cancer_data.corr()['diagnosis'].drop(['diagnosis', 'id'])
top_5_features=corr_matrix.abs().sort_values(ascending=False).head(5).index
grouped_data=cancer_data.groupby('diagnosis', observed=True)[top_5_features].mean()
grouped_data.T.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'orange'])
plt.title('Average Values of Top 5 Features Grouped by Diagnosis')
plt.ylabel('Average Value')
plt.xlabel('Features')
plt.xticks(rotation=45)
plt.legend(title='Diagnosis', labels=['Benign', 'Malignant'])
plt.tight_layout()

#Generating a bar chart that shows the frequency of the target variable for each unique value in a specific feature (e.g., `radius_mean` binned).
cancer_data['radius_mean_bin']=pd.cut(x=cancer_data['radius_mean'], bins=[10, 15, 20, 25, 30])
cross_tab=pd.crosstab(cancer_data['radius_mean_bin'], cancer_data['diagnosis'])
plt.figure(figsize=(10, 6))
cross_tab.plot(kind='bar', stacked=True)

#Comparing My PCA with Scikit Learn's
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler_sklearn = StandardScaler()
pca_sklearn = PCA(n_components = 2)
scaled_cancer_data = scaler_sklearn.fit_transform(cancer_data.select_dtypes(include = [np.number]))
X_pca_sklearn = pca_sklearn.fit_transform(scaled_cancer_data)

pca_scratch = func.PCA(n_components = 2, data = cancer_data)
X_pca_scratch = pca_scratch.fit_transform(cancer_data)

#Explained Variance Ratio
print("Scikit Learn's Explained Variance Ratio: ", pca_sklearn.explained_variance_ratio_)
print("Scratch's Explained Variance Ratio: ", pca_scratch.explained_variance_ratio)

#Eigen Vectors
print("Scikit Learn's Eigen Vectors: ", pca_sklearn.components_)
print("Scratch's Eigen Vectors: ", pca_scratch.selected_components)

#Data Cleaning and Data Preprocessing

data = cancer_data.drop(['diagnosis', 'id'], axis = 1)
target = cancer_data['diagnosis']

#Splitting Train Test Data
splitter = func.train_test_split(test_size = 0.20, random_state = 42, stratified = True)
X_train, X_test, y_train, y_test = splitter.split(X = data, y = target)
print(X_train, X_test, y_train, y_test)
print(np.unique(y_train, return_counts = True))
print(np.unique(y_test, return_counts = True))

#Comparing the splitted data
from sklearn.model_selection import train_test_split
X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(
    data, target, test_size=0.20, random_state=42, stratify=target
)
print(X_train_sk, X_test_sk, y_train_sk, y_test_sk)
print(np.unique(y_train_sk, return_counts = True))
print(np.unique(y_test_sk, return_counts = True))