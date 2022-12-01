import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('OriginalData.csv', index_col=False)
df.head()

x = df.drop(['blueWins'],axis=1)
x.head()

## Dropping gameID feature
df.drop(['gameId'],axis=1,inplace=True)
df.head(2)

## plt displot
for i in df.columns:
    sns.displot(data=df,x=df[i])

col = df.columns

rSkCol = ['blueWardsPlaced','blueWardsDestroyed','redWardsPlaced','redWardsDestroyed']
lSkCol = ['blueAvgLevel','blueTotalExperience','blueTotalMinionsKilled']



# after log1p transformer
for i in rSkCol:
    #before transformer
    sns.displot(data=df,x=df[i])
    #after transformer
    sns.displot(data=df,x=np.log1p(df[i])) 
    
# after square transformer
for i in lSkCol:
    #before transformer
    sns.displot(data=df,x=df[i])
    #after transformer
    sns.displot(data=df,x=np.square(df[i]))    

## calculating corr between feature
corr_matrix = x.corr().abs()
threshold = 0.3
filtered_corr_df = corr_matrix[(corr_matrix >= threshold) & (corr_matrix != 1.000)]
sns.set(rc={"figure.figsize": (100, 100)})
sns.heatmap(filtered_corr_df,annot=True,cmap="Reds")
plt.savefig('HeatMap.png')

# we can see in the imgage that we have multicolinearity therefroe removing those columns
a= {}
for i in x.columns:
    for j in x.columns:
        if(i != j):
            if((abs(np.cov(x[i],x[j]) >=0.4)) and (abs(np.cov(x[i],x[j]))) < 1.0):
                a.add(i)
                a.add(j)




























