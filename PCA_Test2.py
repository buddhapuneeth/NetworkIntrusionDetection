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

def dimReducByPCA(x_train_data,low_dim, doWhiten):
    pca = PCA(n_components=low_dim, whiten=doWhiten)
    return pca.fit(x_train_data).transform(x_train_data)

def doLinearClassification(x_train_data, y_train_data, x_test_data, y_test_data):
    reg = linear_model.LinearRegression()
    reg.fit(x_train_data,y_train_data)
    print(reg.score(x_test_data,y_test_data))

doLinearClassification(x_train,y_train,x_test,y_test)
x_low_dim = dimReducByPCA(df_x,10,'True')
x_train, x_test, y_train, y_test = train_test_split(x_low_dim,df_y,test_size=0.2, random_state=4)
doLinearClassification(x_train,y_train,x_test,y_test)