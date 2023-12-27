
# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix, accuracy_score

# Importing the dataset
dataset = pd.read_csv('creditcard.csv')

# Examine the data structure and available information
print(dataset.head())
print(dataset.describe())
print(dataset.info())
print(dataset.isnull().sum())

# Visualize distributions of features
dataset.hist(figsize=(20, 20))
plt.show()

# Scatter plots for relationships between features
sns.pairplot(dataset, hue='Class')
plt.show()

# Data Preparation
# Feature Scaling
sc = StandardScaler()
dataset['normalizedAmount'] = sc.fit_transform(dataset['Amount'].values.reshape(-1, 1))
dataset = dataset.drop(['Amount', 'Time'], axis=1)

# Splitting the dataset into Features and Target
X = dataset.iloc[:, dataset.columns != 'Class']
y = dataset.iloc[:, dataset.columns == 'Class']

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Clustering for Fraud Detection
kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_train)

# Add the cluster labels to the training features
X_train['Cluster'] = y_kmeans

# Group the data by cluster
print(X_train.groupby('Cluster').mean())

# Check the average fraud rate in each cluster
print(y_train.groupby(X_train['Cluster']).mean())

# PCA for Visualization
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# Plotting the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[y_kmeans == 0, 0], X_train_pca[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X_train_pca[y_kmeans == 1, 0], X_train_pca[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.title('Clusters of transactions')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()

# Classification: Logistic Regression
classifier = LogisticRegression(random_state=42)
classifier.fit(X_train.drop('Cluster', axis=1), y_train.values.ravel())

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Evaluation metrics
print(classification_report(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))


# Clustering Evaluation
silhouette_avg = silhouette_score(X_train.drop('Cluster', axis=1), y_kmeans)
print("For n_clusters = 2 The average silhouette_score is :", silhouette_avg)

# Classification Evaluation
classification_report = classification_report(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Classification Report:\n", classification_report)
print("Confusion Matrix:\n", confusion_matrix)
print("Accuracy:", accuracy)

# Comparative Analysis (Placeholder for your analysis)
print("\nComparative Analysis:")
print("Clustering vs Classification")



