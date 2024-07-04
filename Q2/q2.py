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
#part3
def ruleset1():
    rule=[]
    for i in df['Species']:
        if( i=="setosa" or i=="versicolor" or i=="virginica"):
            rule.append(True)
        else:
            rule.append(False)
    return rule
def ruleset2():
    rule = []
    for i in range(df.shape[0]):
        if (df.iloc[i]["Sepal_Length"] >= 0 and 
            df.iloc[i]["Sepal_Width"] >= 0 and 
            df.iloc[i]["Petal_Length"] >= 0 and 
            df.iloc[i]["Petal_Width"] >= 0):
            rule.append(True)
        else:
            rule.append(False)
    return rule
def ruleset3():
    rule = []
    for i in range(df.shape[0]):
        if df.iloc[i]["Petal_Length"] >= 2 * df.iloc[i]["Petal_Width"]:
            rule.append(True)
        else:
            rule.append(False)
    return rule

def ruleset4():
    rule = []
    for i in range(df.shape[0]):
        if df.iloc[i]["Sepal_Length"] <= 30:
            rule.append(True)
        else:
            rule.append(False)
    return rule

def ruleset5():
    rule = []
    for i in range(df.shape[0]):
        if df.iloc[i]["Sepal_Length"] > df.iloc[i]["Petal_Length"]:
            rule.append(True)
        else:
            rule.append(False)
    return rule

rule1=np.array(ruleset1())
rule2=np.array(ruleset2())
rule3=np.array(ruleset3())
rule4=np.array(ruleset4())
rule5=np.array(ruleset5())
ruledf = pd.DataFrame({
    "Rule1": rule1,
    "Rule2": rule2,
    "Rule3": rule3,
    "Rule4": rule4,
    "Rule5":rule5
})
print("Summary of violations wherein false represents row at which violation occured:\n")
print(ruledf.describe().transpose)
# part4
ruledfint=ruledf.astype(int)
print("Rule violation summary in terms of 0 and 1: \n",ruledfint)
a=plt.figure(1)
plt.bar(["Rule1","Rule2","Rule3","Rule4","Rule5"],[len(rule1)-rule1.sum(),len(rule2)-rule2.sum(),len(rule3)-rule3.sum(),len(rule4)-rule4.sum(),len(rule5)-rule5.sum()])
plt.title("Rule violations representation")
plt.ylim(0,df.shape[0])

#part5
b=plt.figure(2)

plt.boxplot(df2['Sepal_Length'], vert=False, patch_artist=True)
plt.show()

