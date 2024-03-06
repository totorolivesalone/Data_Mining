import pandas as pd 
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,RepeatedKFold,KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
df=pd.read_csv("C:\\Users\\unkno\\Documents\\Data_Mining\\Q5\\iris.csv")
X=df.values[:,:-1]
Y=df.values[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 0) 
k=3
knn=KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,Y_train)
predictions=knn.predict(X_test)
acc=accuracy_score(Y_test, predictions)
print("Accuracy score of KNN Classificaion : ",acc)
conmat= confusion_matrix(Y_test,predictions)
print("Confusion matrix of KNN Classification :",conmat)
