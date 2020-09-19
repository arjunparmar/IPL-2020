#!/usr/bin/env python
# coding: utf-8

# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Loading Dataset
data=pd.read_csv("final_08-19.csv")
data.head()


# Checking Null Values
data.isnull().sum()


# Removing Unwanted Columns
data=data.drop(columns=["id","result","toss_decision","dl_applied","win_by_runs","win_by_wickets","player_of_match","umpire1","umpire2","umpire3","venue"])
data.head()


# Renaming Columns
data = data.rename(columns = {'team A': 'A', 'team B': 'B'}, inplace = False)

print("shape of data is ",data.shape)


# team A=0 team B=1
data.isnull().sum()

# if Team A is winner then it encoded 0 else 1
data.winner[data.winner == data.A] = 0
data.winner[data.winner == data.B] = 1
data.Toss_Winner[data.Toss_Winner == data.A] = 0
data.Toss_Winner[data.Toss_Winner == data.B] = 1
#############################
#This will use in prediction 2020 dataset bcz Label Encoder will encode this way
# data.A[data.A == "CSK"] = 0 
# data.A[data.A == "DC"] = 1
# data.A[data.A == "DEC"] = 2
# data.A[data.A == "GL"] = 3
# data.A[data.A == "KKR"] = 4
# data.A[data.A == "KTK"] = 5
# data.A[data.A == "KXIP"] = 6
# data.A[data.A == "MI"] = 7
# data.A[data.A == "PWI"] = 8
# data.A[data.A == "RCB"] = 9
# data.A[data.A == "RPS"] = 10
# data.A[data.A == "RR"] = 11
# data.A[data.A == "SRH"] = 12
##############################


# In[135]:


print(data.head())


# Encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
data['A']=LE.fit_transform(data['A'])
data['B']=LE.fit_transform(data['B'])
data['Team A Ground']=LE.fit_transform(data['Team A Ground'])
data['Team B Ground']=LE.fit_transform(data['Team B Ground'])
data.head()


# Normalization
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
def Normalization(Params):
    for i in Params:
        data[i]= scaler.fit_transform(pd.DataFrame(data[i])) 
Normalization(["Top Batsmen Points(Team A)","Top Batsmen Points(Team B)","Top Baller Points(Team A)","Top Baller Points(Team B)","runs in pp team a","runs in pp team b","Most Value Player Points(Team A)","Most Value Player Points(Team B)"])


# Our Final Normalized and Encoded Dataset

print(data.head())


# Finding Correlation
# data = data.fillna(method='bfill')
# data.isnull().sum()
# data["top_batsman_ration"] = data["Top Batsmen Points(Team A)"] / data["Top Batsmen Points(Team B)"]
# data["top_boller_ration"] = data["Top Baller Points(Team B)"] / data["Top Baller Points(Team A)"]
# data["ratioA"] = data["Win ratio (Team A)"] * data["Match Won after winning Toss Team A"]
# data["ratioB"] = data["Win ratio (Team B)"] * data["Match Won after winning Toss Team A"]

# data.corr()["winner"]


# Filling Null Values
data = data.fillna(method='bfill')
data.isnull().sum()
data.corr()


# Plotting Heatmap

from sklearn.model_selection import train_test_split
x=data[["A","B","Toss_Winner","Top Batsmen Points(Team A)","Top Batsmen Points(Team B)","Top Baller Points(Team A)","Top Baller Points(Team B)","Match Won after winning Toss Team A","Match Won after winning Toss Team B","runs in pp team a","runs in pp team b","Run Rate(Team A)","Run Rate(Team B)","Most Value Player Points(Team A)","Most Value Player Points(Team B)","Win ratio (Team A)","Win ratio (Team B)","Team A Ground","Team B Ground"]]
y=data['winner']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)



# Random Forest

grid = {
#     "max_leaf_nodes" : list(range(20)),
#     "min_samples_split" : list(range(20))
}

gs = GridSearchCV(
    estimator = RandomForestClassifier(n_estimators=7,max_leaf_nodes=8,random_state=42,n_jobs=-1,verbose=1),
       param_grid =grid ,
    scoring = 'accuracy', 
    n_jobs=-1,
    verbose = 1
)
gs.fit(x_train,y_train)



print("best params for RandomForest ",gs.best_params_)


gs.best_score_

gs.score(x_test,y_test)

# Log Reg

from sklearn.linear_model import LogisticRegression

grid = {
    "max_iter" : list(range(10,100,10)),
    "C": [1,1.2,1.8]     
}

LR = LogisticRegression(
    n_jobs=-1
)

gs_lr = GridSearchCV(
    estimator = LR,
   param_grid =grid ,
    scoring = 'accuracy', 
    n_jobs=-1,
    verbose = 1
)


gs_lr.fit(x_train,y_train)

print("best-score ",gs_lr.best_score_)
gs_lr.best_score_

gs_lr.score(x_test,y_test)


#XGBOOST


get_ipython().system('pip3 install xgboost')
from xgboost import XGBClassifier

gs_xgb = GridSearchCV(
           param_grid = {
               "learning_rate" : [0.01,.1,.08,0.09],
               "subsample" : [.1,.2,.3,.4,.5,.6,.7,.8,.9],
           } ,
        estimator =XGBClassifier(max_depth=3,gamma=0,random=0),    #max_depth = 3 0.08
    scoring = 'accuracy', 
    n_jobs=-1,
    verbose = 1
)


print(gs_xgb.fit(x_train,y_train))



gs_xgb.best_score_


#Support Vector machine

from sklearn.svm import SVC

svm_clf = SVC(C = 50, kernel = 'linear')

# KNN

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=3)

# grid = {
# "leaf_size" : list(range(1,50)),
# "n_neighbors" : list(range(1,30)),
# "p":[1,2]
# }


# gs_knn = GridSearchCV(
#            param_grid = grid ,
#         estimator =KNeighborsClassifier(),    #max_depth = 3 0.08
#     scoring = 'accuracy', 
#     n_jobs=-1,
#     verbose = 1
# )

# print(gs_knn.fit(x_train,y_train))

# Voting Classifier

from sklearn.ensemble import VotingClassifier

v_clf = VotingClassifier(
        estimators = [
            ("lr",gs_lr.best_estimator_),
            ("rf",gs.best_estimator_),
            ("xgb",gs_xgb.best_estimator_), 
            ("svm",svm_clf),  
            ("knn",knn_clf),  
        ],
    voting = 'hard',
    verbose=True,
    n_jobs=-1
    )


v_clf.fit(x_train,y_train)

# v_clf.fit(x_train,y_train)


# In[208]:


# for Prediction

from sklearn import metrics


# gs_xgb
# v_clf

y_pred = gs_xgb.predict(x_test)
Confusion_matrix = print(metrics.confusion_matrix(y_test,y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

# 'learning_rate': 0.01, 'subsample': 0.6}
#  {'gamma': 0, 'learning_rate': 0.08, 'max_depth': 1, 'subsample': 0.5}

# gs_xgb.best_params_


#Prediction of 2020
ipl = pd.read_csv("normalized_2020.csv",index_col=0)
winner = gs_xgb.predict(ipl)


ipl.head()


ipl = pd.read_csv("final_2020.csv")
ipl["winner"] = winner

ipl.winner.value_counts()


ipl = ipl.rename(columns = {'team A': 'A', 'team B': 'B'}, inplace = False)
ipl.winner[ipl.winner == 0] = ipl.A
ipl.winner[ipl.winner == 1] = ipl.B

ipl.winner.value_counts()


# -.MI vs SRH -------------.( KKR Vs CSK ) vs  ( looser of 1st)  


# MI      10
# SRH      9
# CSK      8
# KXIP     7
# RR       7
# KKR      7
# DC       6
# RCB      2

#############################
#This will use in prediction 2020 dataset bcz Label Encoder will encode this way
# data.A[data.A == "CSK"] = 0 
# data.A[data.A == "KKR"] = 4
# data.A[data.A == "MI"] = 7
# data.A[data.A == "SRH"] = 12
##############################

ipl_norm = pd.read_csv("normalized_2020.csv",index_col=0) # team A is already changed with A
ipl = pd.read_csv("final_2020.csv")
ipl = ipl.rename(columns = {'team A': 'A', 'team B': 'B'}, inplace = False)

ipl["Team A Ground"] = "away"
ipl["Team B Ground"] = "away"
ipl_norm["Team A Ground"] = 0 # For Q-F and S-F
ipl_norm["Team A Ground"] = 0

ipl[ipl.A == "SRH"] #48




pd.DataFrame(ipl_norm.loc[[48]]) #KKR vs CSK #  20



gs_xgb.predict(pd.DataFrame(ipl_test.loc[[48]]))   # WINNER KKR


pd.DataFrame(ipl.loc[16]).transpose() #MI VS SRH



winner = gs_xgb.predict(pd.DataFrame(ipl_test.loc[16]).transpose())  #MI vs SRH


winner # ---> SRH




pd.DataFrame(ipl.loc[[31]]) #MI vs KKR





winner = gs_xgb.predict(pd.DataFrame(ipl_test.loc[[31]]))  #MI vs KKR


winner # MI





pd.DataFrame(ipl.loc[55]).transpose() #SRH vs MI




winner = gs_xgb.predict(pd.DataFrame(ipl_test.loc[[55]]))  #SRH vs MI




winner # MI







