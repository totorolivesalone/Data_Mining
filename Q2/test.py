import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("C:\\Users\\unkno\\Documents\\Data_Mining\\Q2\\iris_dirty.csv")
#part 1
df2=df.dropna()
df.drop("Num",axis=1)
print("No. of rows whch are free of null values : ", df2.shape[0])
print("Percentage of rows whch are free of null values : ", (df2.shape[0]/df.shape[0])*100,"%")
#part2
df3=df.replace(to_replace=np.NaN,value="NA")
print(df3.head())
with open("func.txt", "r") as file:
    f = file.read()
exec(f)