# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:19:09 2019

@author: KC
"""

#############################################################################
# --------------------Importing the requied Packages------------
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as stm
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,recall_score,classification_report,r2_score

#############################################################################
    #************** Reading the data into a Pandas DataFrame  ******

fpath = r'C:\Users\KC\Documents\Metro College\DataMining\Project\classification_wine'
fname = r'winequality-red.csv'
file = '{}\{}'.format(fpath,fname)
wine=pd.read_csv(file)
wine.columns

'''
Column List 
'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality'],

''' 

print(os.listdir(fpath))

#############################################################################

#cleaning the data
    #1. check for the nan/NULL values
print(wine.describe())
print(wine.isnull().sum())
#No Null Columns

wine_corr_matrix = wine.corr(method='pearson')
sns.pairplot(wine, kind="reg")

#wine_X= wine.drop('quality',axis=1)
#wine_y=wine['quality']
#sns.pairplot(wine, hue="quality", x_vars=wine_X.columns, y_vars=['quality'], markers=["o", "s", "D","+","*","x"],palette="husl", kind="reg")
#sns.pairplot(wine, x_vars=wine_X.columns, y_vars=['quality'], kind="reg")
#sns.pairplot(wine, x_vars=wine_X.columns, y_vars=['quality'], kind="scatter")
#sns.pairplot(wine, x_vars=wine_X.columns, y_vars=['quality'], kind="scatter")

#--------------------------------------------------------------------------

#Checking for Outliers and removing them

#-------------- defining function to transform  to Z-scores --------

def ztransform(df,column):
    df_col_mean=df[column].mean()
    df_col_std=df[column].std()
    df[column]=(df[column] - df_col_mean) / df_col_std
    return df[column]

wine_s1=wine.copy()
for column in wine_s1.columns:
    print(column)
    wine_s1[column]= ztransform(wine_s1,column) # converting values to Z-Scores

#wine_s1.shape
#wine.shape

sns.boxplot(data=wine_s1).set_title("Before removing outliers")
    

# ------- Function to remove outliers for X baserd on Z-Scores, Z>3
def remove_rows_std_outlier(df,column):
    before_count=df[column].count()
    df.drop(df.index[abs(df[column]) > 3], inplace=True)
    after_count=df[column].count()
    count=before_count-after_count
    print('Removing ouliers in column : {}\nNumber of outliers removed : {}'.format(column,count))

#index_list=wine_s1[abs(wine_s1['fixed acidity']) > 3].index
#wine.drop(wine.index[index_list])

for column in wine_s1.columns:
    if column != 'quality' :
        remove_rows_std_outlier(wine_s1,column)
        

wine_s1.drop('quality',axis=1,inplace=True)      
sns.boxplot(data=wine_s1).set_title("After removing outliers")

#######################################################################################

#--------- Linear Regression using OLS    ---------------

    # --- Preparing the Data For linear Regression ------------
def remove_rows_outlier(df,column):
    df_col_mean=df[column].mean()
    df_col_std=df[column].std()
    df.drop(df.index[abs((df[column] - df_col_mean) / df_col_std) > 3], inplace=True)
    
wine_noout=wine.copy()
for column in wine_noout.columns:
    if column != 'quality' :
        remove_rows_outlier(wine_noout,column)

wine_X = wine_noout.drop(['density','quality'],axis=1)
wine_y = wine_noout['density']

wine_X.columns
    # --- Splitting the Data For Training and Testing ------------
    
X_train, X_test, y_train, y_test = train_test_split(wine_X,wine_y, test_size=0.3, random_state=999)
X_train=stm.add_constant(X_train)
X_test=stm.add_constant(X_test)


#---------- Backward Feature Elimination after setting significance level at 0.05 ----------------

def remove_maxpvalcol(drop_col,X_train,X_test):    
    X_train.drop([drop_col],axis=1,inplace=True)
    X_test.drop([drop_col],axis=1,inplace=True)
i=0
while True: 
    OLS = stm.OLS(y_train,X_train)
    OLSR = OLS.fit()
    OLSR_pval_max=OLSR.pvalues.max()
    i+=1
    if OLSR_pval_max > 0.05:
        drop_col=OLSR.pvalues[OLSR.pvalues==OLSR_pval_max].index[0]
        print('For iteration no : {} \n the max pval is for {} column and the value is {}'.format(i,drop_col,OLSR_pval_max))
        print('Dropping column : {}'.format(drop_col)) 
        remove_maxpvalcol(drop_col,X_train,X_test)
    else:
        print('all the pvalues for the selected explanatory set is <0.05')
        break

print(OLSR.summary())

y_pred = OLSR.predict(X_test)
y_train_pred = OLSR.predict(X_train)


#-------------------  Metrics related to  OLSR ------------

plt.title('Comparison of Y values in test and the Predicted values')
plt.ylabel('Test Set')
plt.xlabel('Predicted values')
plt.scatter(y_test,y_pred, marker ='+')

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 :' , r2_score(y_test,y_pred))
print(OLSR.rsquared)




#sns.pairplot(,x_vars=['fixed acidity', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'pH', 'sulphates','alcohol'], y_vars='density', size=7, aspect=0.7, kind='reg')

######################################################################################


# ---------------------- prepare and run k-fold for Linear Regression --------
wine_X = wine_noout.drop(['density','quality'],axis=1)
wine_y = wine_noout['density']

# - Backward Feature Elimination on entire dataset.
wine_X =stm.add_constant(wine_X)
i=0
while True: 
    OLS = stm.OLS(wine_y,wine_X)
    OLSR = OLS.fit()
    OLSR_pval_max=OLSR.pvalues.max()
    i+=1
    if OLSR_pval_max > 0.05:
        drop_col=OLSR.pvalues[OLSR.pvalues==OLSR_pval_max].index[0]
        print('For iteration no : {} \n the max pval is for {} column and the value is {}'.format(i,drop_col,OLSR_pval_max))
        print('Dropping column : {}'.format(drop_col)) 
        wine_X.drop([drop_col],axis=1,inplace=True)
    else:
        print('all the pvalues for the selected explanatory set is <0.05')
        break
    
OLSR.rsquared

# Training and validating using Kfold where n=10
from sklearn.model_selection import KFold
kf = KFold(n_splits=10,random_state=9, shuffle=False)
wine_X = wine_noout.drop(['density','quality'],axis=1)
wine_y = wine_noout['density']
wine_X =stm.add_constant(wine_X)
r2_tot=0
rsquared_total,n=0,1
for train_index, test_index in kf.split(wine_X):
   #print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = wine_X.iloc[train_index,:], wine_X.iloc[test_index,:]
   y_train, y_test = wine_y.iloc[train_index], wine_y.iloc[test_index]
   OLS = stm.OLS(y_train,X_train)
   OLSR = OLS.fit()
   y_pred = OLSR.predict(X_test)
   rsquared=OLSR.rsquared
   rsquared_total+=rsquared
   rsquared_mean=rsquared_total/n
   r2=r2_score(y_test,y_pred)
   print("Kfold# = {} \t R-Squared = {}      \t R-Squared-mean ={} \t Trained_r2 = {}".format(n,rsquared,rsquared_mean,r2))
   r2_tot+=r2   
   r2_mean=r2_tot/n
   n+=1
   
print(r2_mean)  
print(OLSR.summary())


########################################################################################

# Prepare for Classification Modeling.
#removing free Sulphur dioxide as it looks to have a similar effect as TotalSulphurDioxide.

wine_clean=wine_noout.copy()
wine_clean.drop(['free sulfur dioxide'],axis=1,inplace=True)

level_Rating_3 = []
for i in wine_clean['quality']:
    if i >= 1 and i <= 3:
        level_Rating_3.append('1')
    elif i >= 4 and i <= 7:
        level_Rating_3.append('2')
    elif i >= 8 and i <= 10:
        level_Rating_3.append('3')
wine_clean['LR3'] = level_Rating_3

wine_X= wine_clean.drop(['quality','LR3'],axis=1)
wine_y=wine_clean['LR3']
    
X_train, X_test, y_train, y_test = train_test_split(wine_X, wine_y, random_state = 0)
from sklearn.linear_model import LogisticRegression
def LRfit():
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr.decision_function(X_train)
    global y_pred
    y_pred = lr.predict(X_test)

LRfit()

#create confusion matrix and return accuracy score
def scores(y_test, y_pred):
    print("Confusion Matrix : \n",confusion_matrix(y_test, y_pred))
    print('Accuracy = {}'.format(accuracy_score(y_test, y_pred)*100))
    print('precision = {}'.format(precision_score(y_test, y_pred,average=None)))
    print('sensitivity = {}'.format(recall_score(y_test, y_pred,average=None)))
    print(classification_report(y_test,y_pred))
scores(y_test, y_pred)    

######################################################################################

# prepare and run k-fold for Classification

from sklearn.model_selection import KFold
kf = KFold(n_splits=5,random_state=9, shuffle=False)
total_accuracy,n=0,0
for train_index, test_index in kf.split(wine_X):
   #print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = wine_X.iloc[train_index,:], wine_X.iloc[test_index,:]
   y_train, y_test = wine_y.iloc[train_index], wine_y.iloc[test_index]
   LRfit()
   total_accuracy+=accuracy_score(y_test, y_pred)
   n+=1
   MAC=total_accuracy/n
   print(total_accuracy , MAC)
   


########################################################################################  

#----------- Decision Tree Classifier and K-Fold Implementation ---------------
wine_X= wine_clean.drop(['quality','LR3'],axis=1)
wine_y=wine_clean['LR3']
    
X_train, X_test, y_train, y_test = train_test_split(wine_X, wine_y, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
def DTCfit():
    global dtc_gini
    dtc_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=9, splitter='best')
    dtc_gini.fit(X_train,y_train)
    global y_pred
    y_pred = dtc_gini.predict(X_test)
DTCfit(); 
#print confusion matrix and accuracy score
print(accuracy_score(y_test, y_pred))
sns.scatterplot(X_test, y_pred, color = 'blue')
#prepare k-fold 

from sklearn.model_selection import KFold
kf = KFold(n_splits=10,random_state=9, shuffle=False)
total_accuracy,n=0,0
for train_index, test_index in kf.split(wine_X):
   #print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = wine_X.iloc[train_index,:], wine_X.iloc[test_index,:]
   y_train, y_test = wine_y.iloc[train_index], wine_y.iloc[test_index]
   LRfit()
   Accuracy=accuracy_score(y_test, y_pred)*100
   total_accuracy+=Accuracy
   n+=1
   MAC=total_accuracy/n
   print('K-Fold# =',n)
   #print("Confusion Matrix : \n",confusion_matrix(y_test, y_pred))
   print('Accuracy : {} \t Average_Accuracy : {} '.format(Accuracy,MAC))
   
   
   
#####################################################################################
   
#Visualisation of the tree
from sklearn import tree
tree.plot_tree(dtc_gini) 


plt.subplot()
