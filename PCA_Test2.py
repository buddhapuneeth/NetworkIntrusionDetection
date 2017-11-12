import pandas as pd
import numpy as np 
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.datasets import load_boston
boston = load_boston()
df_x = pd.DataFrame(boston.data, columns=boston.feature_names)
df_y = pd.DataFrame(boston.target)

x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.2, random_state=4)

def dimReduceByPCA(x_train_data,low_dim, doWhiten):
    pca = PCA(n_components=low_dim, whiten=doWhiten)
    return pca.fit(x_train_data).transform(x_train_data)

def dimReduceBySVD(x_train_data,low_dim):
    svd = TruncatedSVD(n_components = low_dim)
    return svd.fit(df_x).transform(df_x)

def doLinearClassification(x_train_data, y_train_data, x_test_data, y_test_data):
    reg = linear_model.LinearRegression()
    reg.fit(x_train_data,y_train_data)
    print(reg.score(x_test_data,y_test_data))

newDim = 2
print("plain")
doLinearClassification(x_train,y_train,x_test,y_test)
x_low_dim = dimReduceByPCA(df_x,newDim,'True')
x_train, x_test, y_train, y_test = train_test_split(x_low_dim,df_y,test_size=0.2, random_state=4)
print("PCA 10")
doLinearClassification(x_train,y_train,x_test,y_test)

print("SVD 10")
x_low_dim_svd = dimReduceBySVD(df_x,newDim)
x_train, x_test, y_train, y_test = train_test_split(x_low_dim_svd,df_y,test_size=0.2, random_state=4)
doLinearClassification(x_train,y_train,x_test,y_test)

print(df_x.shape)
#print(df_x.corr())
df_low_pca = pd.DataFrame(x_low_dim)
print(df_low_pca.shape)
print(df_low_pca.head())
#print(df_low_pca.corr())
# print(x_low_dim.shape)
# print(x_low_dim.corr())

print(np.unique(y_test))