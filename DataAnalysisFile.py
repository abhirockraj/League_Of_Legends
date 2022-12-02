import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('OriginalData.csv', index_col=False)
df.head()

## Dropping gameID feature
df.drop(['gameId'],axis=1,inplace=True)
df.head(2)

x = df.drop(['blueWins'],axis=1)
x.head()

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
a= set()
for i in x.columns:
    for j in x.columns:
        if(i != j):
            b= abs(x[i].corr(x[j]))
            if(b == 1.0):
                     a.add(i)
                     a.add(j)
            
# 7 cols have been removed based on multicolinarity and less corr
X_filter = x.drop(list(a),axis=1)
X_filter.head()
y= df['blueWins']
y.head(2)
full_col = X_filter.columns

rSkCol1,lSkCol1=[],[]
for i in rSkCol:
    if i in full_col:
        rSkCol1.append(i)
for i in lSkCol:
    if i in full_col:
        lSkCol1.append(i)        


# Z score outlier removel
for i in X_filter.columns:
    up = X_filter[i].mean()+3*X_filter[i].std()
    dw = X_filter[i].mean()-3*X_filter[i].std()
    z=X_filter[i]
    X_filter[i]= np.where(X_filter[i] > up,
                   up,
                   np.where(X_filter[i] < dw,
                            dw,
                            X_filter[i]
                           ))

# IQR outlier removel
for i in X_filter.columns:
    up = X_filter[i].quantile(0.75)+1.5*(X_filter[i].quantile(0.75)-X_filter[i].quantile(0.25))
    dw = X_filter[i].quantile(0.25)-1.5*(X_filter[i].quantile(0.75)-X_filter[i].quantile(0.25))
    X_filter[i]= np.where(X_filter[i] > up,
                   up,
                   np.where(X_filter[i] < dw,
                            dw,
                            X_filter[i]
                           ))
    

X_filter.boxplot() 

## feature transformation
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler

transformX = ColumnTransformer(transformers=[('Log Tranform',FunctionTransformer(func=np.log1p),rSkCol1),
                                            ('Square Transform',FunctionTransformer(func=np.square),lSkCol1),
                                           ('Stander Scaler',StandardScaler(),full_col )],remainder='passthrough')

from sklearn.model_selection import train_test_split
X_filter=transformX.fit_transform(X_filter)
xtrain, xtest, ytrain, ytest =  train_test_split( X_filter, y, test_size=0.30, random_state=42)


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

## model building Logistic Reg.
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [1000,100, 10, 1.0, 0.1, 0.01,0.001]
grid = dict(solver=solvers,penalty=penalty,C=c_values)

grid_search = GridSearchCV(estimator=model, param_grid=grid,cv=5, n_jobs=-1, scoring='accuracy')

grid_result = grid_search.fit(xtrain, ytrain)


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


## SVM classifier
from sklearn.svm import SVC
param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

grid_search = GridSearchCV(estimator=SVC(), param_grid=param_grid,cv=5, n_jobs=-1, scoring='accuracy')






















