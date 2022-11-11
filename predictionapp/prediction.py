from django.shortcuts import render
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.metrics import  precision_recall_curve, roc_auc_score, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score,auc, roc_curve, plot_confusion_matrix
import matplotlib.pyplot as plt
color = sns.color_palette()
import pingouin as pg
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score,StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

df=pd.read_csv(r"C:\Users\Sushma\Downloads\lending_club_loan_dataset.csv")

bad_loan_c = pg.pairwise_corr(df, columns=['bad_loan'], method='pearson').loc[:,['X','Y','r']]
bad_loan_c.sort_values(by=['r'], ascending=False)

df["home_ownership"] = df.home_ownership.fillna(df.home_ownership.value_counts().index[0])

df["dti"] = df.dti.fillna(df.dti.mean())

df.drop("id", axis=1, inplace=True)

order={'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}
df['grade']=df['grade'].map(order)

order2={' 36 months':36,' 36 Months':36,' 60 months':60}
df['term']=df['term'].map(order2)

df.replace({'term':{36:0,60:1}},inplace=True)

order3={'credit_card':1,'debt_consolidation':2,'medical':3,'other':4,'home_improvement':5,'small_business':6,'major_purchase':7,
       'vacation':8,'car':9,'house':10,'moving':11,'wedding':12}
df['purpose']=df['purpose'].map(order3)

order4={'RENT':1,'OWN':2,'MORTGAGE':3}
df['home_ownership']=df['home_ownership'].map(order4)


# dummy=pd.get_dummies(df[['purpose','home_ownership']],drop_first=True)

# df=pd.concat([df,dummy],axis=1)
# df=df.drop(['home_ownership','purpose'],axis=1)

X=df.drop('bad_loan',axis=1)
y=df['bad_loan']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)
svm=svm.SVC()
svm.fit(X_train,y_train)

pred=svm.predict(X_test)
accuracy_score(y_test,pred)

lr=LogisticRegression()
lr.fit(X_train,y_train)
# pred1=lr.predict(X_test)
# accuracy_score(y_test,pred1)
#
# rfc=RandomForestClassifier()
# rfc.fit(X_train,y_train)
# pred3=rfc.predict(X_test)
# accuracy_score(y_test,pred3)
#
# k_folds = KFold(n_splits = 10)
# scores = cross_val_score(rfc, X.values, y.values, cv = k_folds)
#
# scores.mean()
#
# sk_folds=StratifiedKFold(n_splits = 10)
# skscores = cross_val_score(rfc, X.values, y.values, cv = sk_folds)
#
# print(skscores.mean())
#
# sc=StandardScaler()
# #ms=MinMaxScaler()
# X=sc.fit_transform(X)


def predictDefault(grade, annualinc, shortemp, emplength, ownershiptype, dti, purpose, term, lastdelinqnone, revolutil, totalreclatefee, odratio):
    print('Entered function')
    try:
        input_data=(int(grade), float(annualinc), int(shortemp), int(emplength), ownershiptype, float(dti), purpose, term, int(lastdelinqnone), float(revolutil), int(totalreclatefee), float(odratio))
        # input_data = (4,78000,0,11,3,18.45,2,0,1,46.3,0,0.0351472)
        input_data_as_numpy_array=np.asarray(input_data)
        input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
        prediction=lr.predict(input_data_reshaped)
    except:
        return 'There was an error. Please try again.'
    if (prediction[0]==0):
        return 'Not Default'
    else:
        return 'Default'

# predictDefault()
