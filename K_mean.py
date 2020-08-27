#Import the relevant libraries
import numpy as np
import pandas as pd

#STARTING WITH THE K-MEAN CLUSTERING METHOD METHOD

#Import Training dataset 
dataset = pd.read_csv('alcohol_data.csv')

#Extracting the Training data
X_train = dataset.iloc[:,[1,2,3]].values
y_train = dataset.iloc[:,5].values

#Importing the test dataset
dataset_test = pd.read_csv('test_data.csv')

#Extracting the test data
X_test = dataset_test.iloc[:,[2,3,4]].values
y_test = dataset_test.iloc[:,5].values

# Feature Scaling (it is done to avoid large influences of one variable on the model if it has larger values)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#encoding the labels to 0 and 1 since ML models need floats or integers to be fed to them
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)

#Clustering using the Kmean clsutering the training dataset
from sklearn.cluster import KMeans
# define the model
model = KMeans(n_clusters=2)
# fit the model
model.fit(X_train)
# assign a cluster to each example
yhat = model.predict(X_train)

#Clustering using the Kmean Clustering the test dataset
from sklearn.cluster import KMeans
# define the model
model = KMeans(n_clusters=2)
# fit the model
model.fit(X_test)
# assign a cluster to each example
yhattest = model.predict(X_test)

#Starting Logistic regression to classify the groups
from sklearn.linear_model import LogisticRegression
classifierLR = LogisticRegression(random_state = 0)
classifierLR.fit(X_train, yhat)

#Predicting the labels for the X_test raw data inputs
y_predLR = classifierLR.predict(X_test)

#Checking the accuracy using Confusion Matrix
from sklearn.metrics import confusion_matrix
cmLR = confusion_matrix(yhattest, y_predLR)
print(cmLR)

#Starting SVM Classifier with rbf kernel
from sklearn.svm import SVC
classifier_SVM = SVC(kernel = 'rbf', random_state = 0)
classifier_SVM.fit(X_train, yhat)

# Predicting the Test set results
y_predSVM = classifier_SVM.predict(X_test)

# Making the Confusion Matrix # This gives the best result
from sklearn.metrics import confusion_matrix
cmSVM = confusion_matrix(yhattest, y_predSVM)
print(cmSVM)

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifierKNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifierKNN.fit(X_train, yhat)

# Predicting the Test set results
y_predKNN = classifierKNN.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yhattest, y_predKNN)
print(cm)
