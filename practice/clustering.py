from sklearn.cluster import DBSCAN,KMeans,AgglomerativeClustering
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=datasets.load_iris()
x=pd.DataFrame(data.data,columns=data.feature_names)
a=plt.figure(1)
plt.scatter(x.values[:,0],x.values[:,1])
plt.title("Scatter plot on actual data")
#kmeans
kmeansmodel=KMeans(n_clusters=3)
kmeansmodel.fit(x)
print("Count of samples in cluster:\n",pd.Series(kmeansmodel.labels_).value_counts())
print("Confusion matrix:",confusion_matrix(data.target,kmeansmodel.labels_))
b=plt.figure(2)
k0=x[kmeansmodel.labels_==0]
k1=x[kmeansmodel.labels_==1]
k2=x[kmeansmodel.labels_==2]
plt.scatter(k0.values[:,0],k0.values[:,1])
plt.scatter(k1.values[:,0],k1.values[:,1])
plt.scatter(k2.values[:,0],k2.values[:,1])
plt.title("KMeans")

##DBSCAN
dbscanmodel=DBSCAN(eps=0.5, min_samples=9)
dbscanmodel.fit(x)
print("Count of samples in cluster:\n",pd.Series(dbscanmodel.labels_).value_counts())
print("Confusion matrix:",confusion_matrix(data.target,dbscanmodel.labels_))
b=plt.figure(3)
d0=x[dbscanmodel.labels_==0]
d1=x[dbscanmodel.labels_==1]
d2=x[dbscanmodel.labels_==2]
plt.scatter(d0.values[:,0],d0.values[:,1])
plt.scatter(d1.values[:,0],d1.values[:,1])
plt.scatter(d2.values[:,0],d2.values[:,1])
plt.title("DBSCAN")
plt.show()

