import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
dataset=[['Milk','Onion','Nutmeg','Kidney Beans','Eggs','Yogurt'],
        ['Dill','Onion','Nutmeg','Kidney Beans','Eggs','Yogurt'],
        ['Milk','Apple','Kidney Beans','Eggs'],
        ['Milk','Unicorn','Corn','Kidney Beans','Yogurt'],
        ['Corn','Onion','Onion','Kidney Beans','Ice cream','Eggs'] 
         ]
te=TransactionEncoder()
te_ary=te.fit(dataset).transform(dataset)
df=pd.DataFrame(te_ary,columns=te.columns_)
### 4.1 minsup=50% minconfidence=75%
print("Minimum support=50% , Minimum confidence=75% \n \n")
frequent_itemsets=apriori(df,min_support=0.5,use_colnames=True)
print("\n \n Frequent Itemsets \n",frequent_itemsets)
from mlxtend.frequent_patterns import association_rules
a=association_rules(frequent_itemsets,metric="confidence",min_threshold=0.75)
print("\n \n \n Association Rules \n \n",a)
### 4.2 minsup=60% minconf=60%
print("Minimum support=60% , Minimum confidence=60% \n \n")
frequent_itemsets2=apriori(df,min_support=0.6,use_colnames=True)
print("\n \n Frequent Itemsets \n",frequent_itemsets2)
from mlxtend.frequent_patterns import association_rules
a2=association_rules(frequent_itemsets,metric="confidence",min_threshold=0.6)
print("\n \n \n Association Rules \n \n",a2)

