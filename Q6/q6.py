from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
iris = datasets.load_iris ()
x = iris.data[:, :4] #means we take 4 dimension in the feature space
#plot data distribution
plt.scatter(x[:, 0], x[:, 1], c="red", marker="o")
plt.title("ORIGINAL DATA SAMPLE")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()
#****SIMPLE KMEANS****
estimator = KMeans(n_clusters=3) #construct a clusterer
estimator.fit(x) #clustering
label_pred = estimator.labels_ #get cluster labels
print("Number of samples per cluster in Simple K-Means Clustering:")
print(pd.Series(estimator.labels_).value_counts())
print(confusion_matrix(iris.target, estimator.labels_))
#Draw k-means results
x0 = x[label_pred ==0]
x1 = x[label_pred == 1]
x2 = x[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker="o", label="Setosa") 
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker="*", label="Versicolor")
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker="+", label="Virginica")
plt.title("SIMPLE K-MEANS CLUSTERING")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend (loc=2)
plt.show()
#****DBSCAN****
dbscan = DBSCAN(eps=0.5, min_samples=9)
dbscan.fit(x)
label_pred = dbscan.labels_
print("Number of samples per cluster in DBSCAN Clustering:")
print(pd.Series(dbscan.labels_).value_counts ())
print("DBSCAN Clustering Result:")
print(confusion_matrix(iris.target, dbscan.labels_))
#Draw k-means results
x0 = x[label_pred == 0]
x1 = x[label_pred == 1]
x2 = x[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker="o", label="Setosa")
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker="*", label="Versicolor")
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker="+", label="Virginica")
plt.title("DBSCAN")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend (loc=2)
plt.show()
#****HIREECHICAL ****

irisdata = iris.data
clustering = AgglomerativeClustering(linkage="ward", n_clusters=3)
res = clustering.fit (irisdata)
print("Number of samples per cluster in Hirerachical Clustering:")
print(pd.Series(clustering.labels_).value_counts())
print("Hirerachical Clustering Result:")
print(confusion_matrix(iris.target, clustering.labels_))
plt.figure()
d0 = irisdata [clustering.labels_ == 0]
plt.plot(d0[:, 1], d0[:,1], "r.")
d1 = irisdata [clustering.labels_ == 1]
plt.plot (d1[:,0], d1[:,1], "go")
d2 = irisdata [clustering.labels_ ==2]
plt.plot (d2[:, 0], d2[:,1], "b *")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("HIRERCHICAL (AGNES) CLUSTERING")
plt.show()
