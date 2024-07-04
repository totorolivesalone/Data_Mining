import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

# Load the wine dataset
wine = load_wine()
df = pd.DataFrame(data=wine.data,columns=wine.feature_names)
# Check the mean and standard deviation of each attribute before standardization
means_before = df.mean()
stds_before = df.std()

print("Means of the attributes before standardization:\n", means_before)
print("Standard deviations of the attributes before standardization:\n", stds_before)

# Standardize the dataset
scaler = StandardScaler()
X_standardized = scaler.fit_transform(df)
print("After standardization::")
df2=pd.DataFrame(X_standardized,columns=wine.feature_names)
print(df2.head())
print(df2.mean())
print(df2.std())