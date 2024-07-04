import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,RepeatedKFold, KFold
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
ds = pd.read_csv('iris.csv')
X = ds.values[:, :-1]
Y = ds.values[:, -1]
print(X.shape)
print(Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X,
Y, test_size=0.3, random_state=3)
#Naive Bayes Classifier
print("----------------NAIVE BAYES' CLASSIFIER \n")
NBclassifier = GaussianNB()
NBclassifier.fit(X_train, Y_train)
predictions = NBclassifier.predict(X_test)
#print("Predicted response of NAIVE BAYES CLASSIFIER:\n", predictions)
#Accuracy Score
acc= accuracy_score(Y_test, predictions)
print("Accuracy score of NAIVE BAYES CLASSIFIER: ",
acc)
# # Confusion Matrix
conmat= confusion_matrix(Y_test, predictions)
print("CONFUSION MATRIX of Y_Test and Predictions: \n",
conmat)
#Decision Tree Classifier
print("\n----------------DECISION TREE CLASSIFIER \n")
DTclassifer= DecisionTreeClassifier()
DTclassifer.fit(X_train, Y_train)
predictions1 = DTclassifer.predict(X_test)
#print("Predicted response of DECISION TREE CLASSIFIER:\n", predictions1)
#Accuracy Score
acc1= accuracy_score(Y_test, predictions1)
print("Accuracy score of DECISION TREE CLASSIFIER: ",
acc1)
# # Confusion Matrix
conmat1= confusion_matrix(Y_test, predictions1)
print("CONFUSION MATRIX of Y_Test and Predictions: \n",
conmat1)
#K- Nearest Neighbours
print("\n----------------K-NEAREST NEIGHBOUR \n")
# KNN model requires you to specify n_neighbors,the number of points the classifier will look at to determine what class a new point belongs to
KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_train, Y_train)
predictions2 = KNN_model.predict(X_test)
#print("Predicted response of KNN: \n", predictions2)
#Accuracy Score
acc2= accuracy_score(Y_test, predictions1)
print("Accuracy score of KNN: ", acc2)
#onfusion Matrix
conmat2= confusion_matrix(Y_test, predictions2)
print("CONFUSION MATRIX of Y_Test and Predictions: \n", conmat2)
print("\n----------------TRAINING SET SPLITTING USING HOLDOUT METHOD \n")
Val_size = 0.25 # test size is how much rest is training
random_seed = 3 # randomly choosing
X_train, X_test, Y_train, Y_test = train_test_split(X,
Y, test_size=Val_size,
random_state=random_seed)
deciTree = DecisionTreeClassifier()
deciTree.fit(X_train, Y_train)
predictions3 = deciTree.predict(X_test)
print("Accuracy on the Test Data when split is 0.25 using DECISION TREE")
print(accuracy_score(Y_test, predictions3))
Val_size = 0.33 # test size is how much rest is
X_train, X_test, Y_train, Y_test = train_test_split(X,
Y, test_size=Val_size,
random_state=random_seed)
deciTree = DecisionTreeClassifier()
deciTree.fit(X_train, Y_train)
predictions4 = deciTree.predict(X_test)
print("Accuracy on the Test Data when split is 0.33 using DECISION TREE")
print(accuracy_score(Y_test, predictions4))
print("\n----------------TRAINING USING RANDOM FOREST CLASSIFIER \n")
# Performing training
Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y,
test_size = 0.3, random_state = 100)
clfr= RandomForestClassifier(random_state = 100)
clfr.fit(Xtrain, ytrain)
predictions5 = clfr.predict(Xtest)
print("Accuracy on the Test Data using RANDOM FOREST CLASSIFIER")
print(accuracy_score(ytest, predictions5))
print("\n----------------TRAINING SET BY K-FOLD CROSS VALIDATION \n")
kf = KFold(n_splits=5, random_state=None,
shuffle=False)
for train_index, test_index in kf.split(X):
    X_train1, X_test1 = X[train_index], X[test_index]
    Y_train1, Y_test1 = Y[train_index], Y[test_index]
deciTree = DecisionTreeClassifier()
deciTree.fit(X_train1, Y_train1)
predictions = deciTree.predict(X_test1)
print("Accuracy on the Test Data AFTER CROSS VALIDATION")
print(accuracy_score(Y_test1, predictions))
print("\n----------------Repeated KFold \n")
#Scaling of data using minmaxscaler()-NORMALIZATION
rkf=RepeatedKFold(n_splits=5,random_state=None)
for train_index, test_index in rkf.split(X):
    X_train2, X_test2 = X[train_index], X[test_index]
    Y_train2, Y_test2 = Y[train_index], Y[test_index]
deciTree = DecisionTreeClassifier()
deciTree.fit(X_train2, Y_train2)
predictions = deciTree.predict(X_test1)
print("Accuracy on the Test Data AFTER CROSS VALIDATION")
print(accuracy_score(Y_test1, predictions))
