import cPickle
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
#dataset a predecir
df = pd.read_csv('test.csv')

numerical_features = ['engine_id', 'flight_id', 't_1', 't_2', 't_3', 't_4', 't_oil', 'p_oil', 'vibrations_2', 'vibrations_4', 'core_speed', 'fan_speed', 'thrust']

def remove_strings(features):
    for feature in features:
        df[feature].replace('[a-zA-Z_ ]+', np.nan, inplace=True, regex=True)
    df.dropna(axis=0, inplace=True)

remove_strings(numerical_features)



badtypes = ['t_3','t_4','vibrations_2','vibrations_4','core_speed']
df[badtypes]= df[badtypes].astype(float)


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

df.dropna(axis=0, inplace=True)


ag = df.groupby('engine_id').mean()
with open('modelLinear.pkl', 'rb') as fid:
    modelLinear = cPickle.load(fid)
with open('modelKNN.pkl', 'rb') as fid:
    modelKNN = cPickle.load(fid)
with open('modelLogistic.pkl', 'rb') as fid:
    modelLogistic = cPickle.load(fid)


features = ['core_speed', 'fan_speed']

predictedKNN = modelKNN.predict(ag[features])
predictedLinear = modelLinear.predict(ag[features])
predictedLogistic =modelLogistic.predict(ag[features])
category = []
for value in predictedLinear:
        if value < 50:
            category.append(0)
        else:
            category.append(1)
ag['category-KNN'] = predictedKNN
ag['category-Logistic'] = predictedLogistic
ag['category-Linear'] = category
ag['damage-Linear'] = predictedLinear
ag.to_csv('PREDICCION.csv')
