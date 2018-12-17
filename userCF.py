import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.stats import pearsonr


topN=50
data=pd.read_csv('data.csv')
data=data.iloc[:100,:]
print(data.head())

bina=preprocessing.binarize(data,threshold=0.0, copy=True)
bina=pd.DataFrame(data=bina,columns=data.columns)
print(bina.head())
print(data.head())


#z-score
for i in range(data.shape[0]):
    tmplist=[]
    for j in data.columns:
        if bina.loc[i,j]:
            tmplist.append(data.loc[i,j])
    t=preprocessing.scale(tmplist)
    index=0
    for j in data.columns:
        if bina.loc[i,j]:
            data.loc[i,j]=tmplist[index]
            index+=1

#new user
test=np.random.randint(0,100,(1,20))[0]
print(test)
print(data.columns.values.tolist())
test=pd.Series(test, index=data.columns.values.tolist())
#test_bin=pd.Series(preprocessing.binarize(test,threshold=0.0, copy=True),index=data.columns.values.tolist())
test_bin=[]
for i in test.values:
    if i ==0:
        test_bin.append(0)
    else:
        test_bin.append(1)
test_bin=pd.Series(test_bin,index=data.columns.values.tolist())


#topN nearest neighborhoods
pearson=[]
for i in range(data.shape[0]):
    a,u=[],[]
    for j in data.columns:
        if test_bin[j]==1 and bina.loc[i,j]==1:
            a.append(test[j])
            u.append(data.loc[i,j])
    p=pearsonr(a,u)[0]
    pearson.append([p,i])
pearson=sorted(pearson,reverse=True)
print(pearson)
pearson=pearson[:topN]


#recommender
recom=[]
for j in data.columns:
    score,w=0.0,0.0
    for t in pearson:
        p,i=t[0],t[1]
        if bina.loc[i,j]==1:
            score,w=score+data.loc[i,j]*p,w+p
    recom.append([score/w,j])

recom=sorted(recom,reverse=True)

print('Recommender results:')
print(recom)









