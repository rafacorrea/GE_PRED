# -*- coding: utf-8 -*-
"""
Created on Tue May  3 19:26:55 2016

@author: JacoboRaye
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 19:22:14 2016

@author: JacoboRaye
"""

#%% Librerias

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from scipy.special import expit
from sklearn import tree 
from sklearn.metrics import precision_recall_fscore_support
from sklearn.externals.six import StringIO
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
import pylab as pl
import statsmodels.api as sm
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import metrics
import pydot_ng


#%% 1-Data Cleansing: Identify data types and remove any invalid values and outliers in the data set for each variable. An invalid value could be a number out of range or an incorrect input
     
     # Cargar csv
df = pd.read_csv('data.csv')

    # Tipo de datos
df.dtypes
df.head(10)
len1 = len(df) 

#Reemplazar los strings por NA 
df['engine_id'].replace('[a-zA-Z_ ]+', np.nan, inplace=True, regex=True)    
df['flight_id'].replace('[a-zA-Z_ ]+', np.nan, inplace=True, regex=True)
df['t_1'].replace('[a-zA-Z_ ]+', np.nan, inplace=True, regex=True)
df['t_2'].replace('[a-zA-Z_ ]+', np.nan, inplace=True, regex=True)
df['t_3'].replace('[a-zA-Z_ ]+', np.nan, inplace=True, regex=True)                    
df['t_4'].replace('[a-zA-Z_ ]+', np.nan, inplace=True, regex=True)                    
df['t_oil'].replace('[a-zA-Z_ ]+', np.nan, inplace=True, regex=True)                    
df['p_oil'].replace('[a-zA-Z_ ]+', np.nan, inplace=True, regex=True)  
df['vibrations_2'].replace('[a-zA-Z_ ]+', np.nan, inplace=True, regex=True)
df['vibrations_4'].replace('[a-zA-Z_ ]+', np.nan, inplace=True, regex=True)
df['core_speed'].replace('[a-zA-Z_ ]+', np.nan, inplace=True, regex=True)
df['fan_speed'].replace('[a-zA-Z_ ]+', np.nan, inplace=True, regex=True)
df['thrust'].replace('[a-zA-Z_ ]+', np.nan, inplace=True, regex=True)


#Cambiamos los types que son Objects por Floats
badtypes = ['t_3','t_4','vibrations_2','vibrations_4','core_speed']
df[badtypes]= df[badtypes].astype(float)


#Reemplazamos las presiones, vibraciones y velocidades negativas así como las temperaturas menores a -273 ºC
df.loc[df['p_oil'] < 0] = np.nan
df.loc[df['t_1'] < -273.15] = np.nan
df.loc[df['t_2'] < -273.15] = np.nan
df.loc[df['t_3'] < -273.15] = np.nan
df.loc[df['t_4'] < -273.15] = np.nan
df.loc[df['t_oil'] < -273.15] = np.nan
df.loc[df['vibrations_2'] < 0] = np.nan
df.loc[df['vibrations_4'] < 0] = np.nan
df.loc[df['thrust'] < 0] = np.nan
df.loc[df['core_speed'] < 0] = np.nan
df.loc[df['fan_speed'] < 0] = np.nan


#Quitamos todo lo reemplazado anteriormente
df.dropna(axis=0, inplace=True)



df.dtypes

len2 = len(df) 
dif = len1 - len2
print '%d rows were removed' % dif


#%% Aggregate the data by engine_id by computing the mean of each parameter.

ag = df.groupby('engine_id').mean()
agc = ag
ag



#Calculate the number of flights for each engine and include it to the aggregated dataset.
ag['number_flights']=df[['engine_id','flight_id']].groupby('engine_id').count() 
ag.drop(['flight_id'], axis=1, inplace=True)

ag

list = []
for engine in ag.index.values:
    list.append(df.loc[df['engine_id'] == engine]['customer'].unique()[0])

ag['customer'] = list


list = []
for engine in ag.index.values:
    list.append(df.loc[df['engine_id'] == engine]['engine_type'].unique()[0])

ag['engine_type'] = list

ag
#%%Pasar la cateogria a binario 
df['category'][df['category'] == 'failed'] = 1
df['category'][df['category'] == 'non-failed'] = 0
df['category'] = df['category'].astype(int)
ag['category']=df[['engine_id', 'category']].groupby('engine_id').mean()
ag
ag.to_csv('LIMPIO.csv')

ff =ag.loc[ag['category']== 1] 
ff=len(ff)
print 'There are %d engine_ids that failed' %ff


#%%Understand the data, how many different engine_id, engine_type, customer and total flights are there?

ag[['category','engine_type']].groupby('engine_type').count() 
ag.loc[ag['category']== 1 ] 

ag[['customer','category']].groupby('category').count() 

en =len(df[['engine_id']].groupby('engine_id').count())
print 'There are %d different engine_id`s' %en



fn = len(df[['flight_id']].groupby('flight_id').count())
print 'There are %d different flight_id`s' %fn

df[['engine_type','engine_id']].groupby('engine_type').count() 

df[['customer','damage']].groupby('customer').mean() 

df.loc[df['flight_id']== 6900 ] 


#%%Análisis por customer de los motores fallidos 

dme = ag.loc[ag['customer']== 'DME'] 
dme = dme.loc[dme['category']==1]
dme = dme.loc[dme['engine_type']=='EX-50A']
dme =len(dme)
print 'There are %d engine`s from customer DME and engine type EX-50A that failed' %dme

asi = ag.loc[ag['customer']== 'ASI'] 
asi = asi.loc[asi['category']==1]
asi = asi.loc[asi['engine_type']=='EX-50B']
asi =len(asi)
print 'There are %d engine`s from customer ASI and engine type EX-50B that failed' %asi

#%% Statistical summary 
df.describe()





#Pie charts
x = df.groupby('customer').count()['flight_id']
labels = x.index.values

plt.pie(x, labels = labels, autopct='%1.1f%%')
plt.savefig('customer_share.png')
plt.close()


x = df.groupby('engine_type').count()['flight_id']
labels = x.index.values

plt.pie(x, labels = labels, autopct='%1.1f%%')
plt.savefig('type_share.png')
plt.close()


#Boxplots
features = ['t_1','t_2', 't_3', 't_4', 't_oil', 'p_oil', 'vibrations_2' , 'vibrations_4', 'core_speed', 'fan_speed', 'damage', 'thrust']



    
def boxplot(features):
    for customer in ag['customer'].unique():
        temp = ag.loc[ag['customer'] == customer]
        normalized = temp[features].apply(lambda x: (x-x.min()) / (x.max() - x.min()))
        normalized.boxplot(vert = False)
        plt.savefig('boxplot_%s.png' % customer)
        plt.close()

boxplot(features)

ag.hist()
pl.show()

#%% Scatter and density plots 
#numerical_features = ['engine_id', 'flight_id','damage', 'p_oil', 't_oil', 't_1', 't_2', 't_3','t_4','vibrations_2','vibrations_4','core_speed', 'fan_speed', 'thrust']
from scipy.stats import gaussian_kde
import scipy.misc

def scatter(feature):
    failed = ag.loc[ag['category'] == 1]
    nonfailed = ag.loc[ag['category'] == 0]
    plt1 = plt.scatter(failed[feature], failed['damage'], color='Red')
    plt2 = plt.scatter(nonfailed[feature], nonfailed['damage'], color='Blue')
    
    plt.legend((plt1, plt2), ('failed', 'non-failed'), loc = 'lower right', ncol=1)
    plt.xlabel(feature)
    plt.ylabel('damage')
    plt.title('%s vs damage' % feature)
    plt.savefig('%s-damage_scatter.png' % feature)
    #plt.show()
    plt.close()
    
    
def density(feature):
    y = ag['damage']
    x = ag[feature]
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    #fig, ax = plt.subplots()
    plt.scatter(x, y, c=z, s=50, edgecolor='')
    plt.xlabel(feature)
    plt.ylabel('damage')
    plt.title('%s vs damage' % feature)    
    cb = plt.colorbar()
    cb.set_label('densidad')
    plt.savefig('%s-damage_density.png' % feature)
    #plt.show()
    plt.close()

def histograma(feature, divisiones):
    data = df[feature]
    plt.hist(data, divisiones)
    plt.title('histograma de %s (%d divs)' % (feature, divisiones))
    plt.savefig('%s-histograma.png' % feature)
    plt.close()
    

features=['t_1', 't_2', 't_3', 't_4', 't_oil', 'p_oil', 'vibrations_2', 'vibrations_4', 'core_speed', 'fan_speed', 'thrust']


for feature in features:
    density(feature)
    scatter(feature)
    histograma(feature, 100)
    

#%%Compute correlations matrix for all variables. Which is the most correlated pair of variables?
    
agc = ag.corr()
agc.to_csv('Correlaciones.csv')

#List all the variables and provide the R-squared value with respect to the Y for each one. Order them using R-squared (highest at the top).
from scipy import stats
list = ['t_1','t_2', 't_3', 't_4', 't_oil', 'p_oil', 'vibrations_2' , 'vibrations_4', 'core_speed', 'fan_speed', 'thrust']
y = ag['damage']
results =[]
r_values = []
for x in list: 
    slope, intercept, r_value, p_value, std_err = stats.linregress(ag[x],y)
    #print "%s"%x, r_value**2
    results.append([r_value**2, x])
    r_values.append([r_value, x])

#ordena de mayor a menor
results.sort(reverse=True)
r_values.sort(reverse=True)
print 'R^2 values %s\n' % results
print 'R valyes %s\n' % r_values



#Modelo y KFold
confusion = 0
accuracy = 0
folds = 10

from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from subprocess import check_call


#KNN
confusion = 0
accuracy=0
features = ['core_speed', 'fan_speed']
x = ag[features]
y = ag['category']

skf = StratifiedKFold(y, n_folds=folds)

for train_index, test_index in skf:
    modelKNN = KNeighborsClassifier()
    modelKNN.fit(x.iloc[train_index], y.iloc[train_index])
    predicted = modelKNN.predict(x.iloc[test_index])
    expected = y.iloc[test_index]
    confusion = np.add(metrics.confusion_matrix(expected, predicted), confusion)
    accuracy += metrics.accuracy_score(expected, predicted)
    

#suma de las matrices de confusion del KFold
print 'Matriz de confusion:\n %s' % confusion

#calcular la precision del modelo
print 'Precision: %f%%' % ((accuracy/folds) * 100)


#REGRESION LINEAL
accuracy=0
features = ['core_speed', 'fan_speed']
x = ag[features]
y = ag['damage']

skf = StratifiedKFold(y, n_folds=folds)

iteracion = 0

for train_index, test_index in skf:
    modelLinear = LinearRegression()
    modelLinear.fit(x.iloc[train_index], y.iloc[train_index])
    iteracion += 1
    predicted = modelLinear.predict(x.iloc[test_index])
    expected = y.iloc[test_index]
    lista=[]
    for value in predicted:
        if value < 50:
            lista.append(0)
        else:
            lista.append(1)
    predicted = lista
    lista=[]
    for value in expected:
        if value < 50:
            lista.append(0)
        else:
            lista.append(1)
    expected = lista
    accuracy += metrics.accuracy_score(expected, predicted)
    

#calcular la precision del modelo
print 'Precision: %f%%' % ((accuracy/folds) * 100)



#REGRESION LOGISTICA
confusion = 0
accuracy=0
features = ['core_speed', 'fan_speed']
x = ag[features]
y = ag['category']

skf = StratifiedKFold(y, n_folds=folds)

for train_index, test_index in skf:
    modelLogistic = LogisticRegression(class_weight = "balanced")
    modelLogistic.fit(x.iloc[train_index], y.iloc[train_index])
    predicted = modelLogistic.predict(x.iloc[test_index])
    expected = y.iloc[test_index]
    confusion = np.add(metrics.confusion_matrix(expected, predicted), confusion)
    accuracy += metrics.accuracy_score(expected, predicted)
    



#suma de las matrices de confusion del KFold
print 'Matriz de confusion:\n %s' % confusion

#calcular la precision del modelo
print 'Precision: %f%%' % ((accuracy/folds) * 100)

#categorical
x = ag[features]
y = ag['category']
modelKNN.fit(x, y)
modelLogistic.fit(x,y)

#non-categorical
y = ag['damage']
modelLinear.fit(x,y)


import cPickle
with open('modelKNN.pkl', 'wb') as fid:
    cPickle.dump(modelKNN, fid)  
with open('modelLinear.pkl', 'wb') as fid:
    cPickle.dump(modelLinear, fid)  
with open('modelLogistic.pkl', 'wb') as fid:
    cPickle.dump(modelLogistic, fid)  
                              
# Read and prepare test data
