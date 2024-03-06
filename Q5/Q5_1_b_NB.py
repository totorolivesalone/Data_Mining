import pandas as pd 
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split,RepeatedKFold,KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
df=pd.read_csv("C:\\Users\\unkno\\Documents\\Data_Mining\\Q5\\iris.csv")
X=df.values[:,:-1]
Y=df.values[:,-1]

print ("Dataset is : \n",df)
print (X.shape)
print (Y.shape)
# X_train, X_test, Y_test = train_test_split(X,Y,test_size=0.3 random_state=3)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=3)
#NB Classifier
NBclassifier= GaussianNB() # this will contains the pattrns learned by training datasets

NBclassifier.fit(X_train,Y_train)

predictions= NBclassifier.predict(X_test)
acc=accuracy_score(Y_test, predictions)
print("Accuracy score of NB Classificaion : ",acc)
conmat= confusion_matrix(Y_test,predictions)
print("Confusion matrix of NB Classification :",conmat)