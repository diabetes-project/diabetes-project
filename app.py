#import libraries
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

#Initialize the flask App
app = Flask(__name__)
ohe, ss, model = pickle.load(open('model_diabetes.pkl', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    ls = [x for x in request.form.values()]
    # int_features = []
    # for i in ls:
    #     try:
    #         u = int(i)
    #         int_features.append(u)
    #     except:
    #         int_features.append(i)
    # catcol = ohe.transform(pd.DataFrame(int_features[:7]).T).A
    # numcol = ss.transform(pd.DataFrame(int_features[7:]).T)
    # x_train = np.concatenate([catcol, numcol], axis=1)
    cols = ['Ky_QT_adjust','hang','GIOI_TINH','nhomTuoi','Tuổi','HI','insurance','regions','checkComorbidities','comorbidities','soBienChung_cat',
              'loaiBienChung','diemBienChung_cat','tuyen','MA_LYDO_VVIEN','MA_LOAI_KCB','Số ngày điều trị']
    x = pd.DataFrame([ls], columns=cols)
    cols_1 = ['Ky_QT_adjust','GIOI_TINH','Tuổi','nhomTuoi','regions','HI','insurance','hang','tuyen','MA_LYDO_VVIEN','MA_LOAI_KCB','Số ngày điều trị','checkComorbidities',
            'comorbidities','soBienChung_cat','loaiBienChung','diemBienChung_cat']
    x = x[cols_1]
    # return ls
    prediction = pipeline.predict(x)

    output = int(np.exp(prediction[0]))

    return render_template('index.html', prediction_cost='Chi phí điều trị một năm của bệnh nhân: {:,} VND'.format(output).replace(',','.'))
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=1003)
