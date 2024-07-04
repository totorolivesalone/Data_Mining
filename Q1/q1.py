import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("people.csv")
print(df)
def ruleset1():
    rule=[]
    for i in df['Age']:
        if(i>=0 and i<=150):
            rule.append(True)
        else:
            rule.append(False)
    return rule
def ruleset2():
    rule=[]
    for i in range(0,df.shape[0]):
        if df.at[i,"Age"]> df.at[i,"yearsmarried"]:
            rule.append(True)
        else:
            rule.append(False)
    return rule
def ruleset3():
    rule=[]
    stat=["single","married","widowed"]
    for i in df["status"]:
        if(i in stat):
            rule.append(True)
        else:
            rule.append(False)
    return rule
def ruleset4():
    rule=[]
    for i in range(0,df.shape[0]):
        if(df.at[i,"Age"]>=0 and df.at[i,"Age"]<18 and df.at[i,"agegroup"]=="child"):
            rule.append(True)
        elif(df.at[i,"Age"]>=18 and df.at[i,"Age"]<65 and df.at[i,"agegroup"]=="adult"):
            rule.append(True)
        elif(df.at[i,"Age"]>=65 and df.at[i,"Age"]<=150 and df.at[i,"agegroup"]=="elderly"):
            rule.append(True)
        else:
            rule.append(False)
    return rule
rule1=np.array(ruleset1())
rule2=np.array(ruleset2())
rule3=np.array(ruleset3())
rule4=np.array(ruleset4())
print("Violations found::\n")
if(False in rule1):
    print("Invalid age data")
if(False in rule2):
    print("Inconsistent relationship between age and yearsmarried feature data")
if(False in rule3):
    print("Invalid data in status feature")
if(False in rule4):
    print("Inconsistent relationship between age and agegroup feature data\n")
ruledf = pd.DataFrame({
    "Rule1": rule1,
    "Rule2": rule2,
    "Rule3": rule3,
    "Rule4": rule4
})
print("Summary of violations wherein false represents row at which violation occured:\n")
print(ruledf)
ruledfint=ruledf.astype(int)
print("Rule violation summary in terms of 0 and 1: \n",ruledfint)
plt.bar(["Rule1","Rule2","Rule3","Rule4"],[len(rule1)-rule1.sum(),len(rule2)-rule2.sum(),len(rule3)-rule3.sum(),len(rule4)-rule4.sum()])
plt.title("Rule violations representation")
plt.ylim(0,df.shape[0])
plt.show()

