
'''
This is a MLP model to predict the annual cost per patient according to personal characteristics
Dataset:
diabetes.csv, which contains more than 2 millions records of patients 
from Vietnam Social Security data

'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor

import pickle

# Prepare data

# Load data
df = pd.read_csv('https://media.githubusercontent.com/media/dothanhdatle/health_cost_prediction/huy/project/diabetes.csv?token=AUAF5NXBFUJYLCSQJ7A22T3FMMYWU', sep = ',')

# Change data types
df[['GIOI_TINH','checkComorbidities','insurance','checkBienChung']] = df[['GIOI_TINH','checkComorbidities','insurance','checkBienChung']].astype(object)

# Create log_TongChi
df['log_TongChi'] = np.log(df['TongChi'])

# Drop uneccessary columns
df.drop(['type','nhomTuoi','TongChi', 'MA_THE','BHTT', 'BNTT', 'BNCCT', 'BNT', 'XN', 'CDHA', 'Thuoc', 'Mau',
       'TTPT', 'VTYT', 'DVKT', 'Kham', 'Giuong', 'VChuyen', 'NguonKhac'], axis=1, inplace=True)


# Preprocessing
x_train, x_test, y_train, y_test = train_test_split(df.drop('log_TongChi', axis=1), df.log_TongChi, test_size=0.2, stratify=df['Ky_QT_adjust'])
ohe = OneHotEncoder()
ss = StandardScaler()


# Training model

def train(x_train, y_train):
    cat_train = ohe.fit_transform(x_train.select_dtypes('object')).A
    num_train = ss.fit_transform(x_train.select_dtypes(['int64','float64']))
    X_train = np.concatenate([cat_train, num_train], axis=1)

    model = MLPRegressor(hidden_layer_sizes=(25), activation='relu', solver='adam', alpha=0.001, batch_size=128, learning_rate='adaptive')
    model.fit(X_train, y_train)

    return ohe, ss, model

def predict(x_test, y_test, ohe, ss, model):
    cat_test = ohe.transform([x_test.select_dtypes('object')]).A
    num_test = ss.transform([x_test.select_dtypes(['int64','float64'])])
    X_test = np.concatenate([cat_test, num_test], axis=1)

    result = model.predict(X_test)
    return result

ohe, ss, model = train(x_train, y_train)
y_pred = predict(x_test, y_test, ohe, ss, model)

print(y_pred)
# Saving model
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump((ohe, ss, model), open('model_diabetes.pkl','wb'))

'''
#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2.6, 8, 10.1]]))
'''
