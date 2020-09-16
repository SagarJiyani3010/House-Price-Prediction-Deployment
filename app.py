import numpy as np
from flask import Flask,request, url_for, redirect, render_template, jsonify
import pickle
import pandas as pd
from pycaret.regression import *

app = Flask(__name__)
model=load_model('final_model')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features=[x for x in request.form.values()]
    final=np.array(int_features)
    col =['MSZoning', 'LotArea', 'Utilities', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'RoofStyle', 'BsmtQual', 'BsmtCond', 'CentralAir', 'KitchenQual', 'GarageType', 'GarageArea', 'GarageQual', 'GarageCond']
    data_unseen = pd.DataFrame([final], columns = col)
    print(int_features)
    prediction=predict_model(model, data=data_unseen, round = 0)
    prediction=int(prediction.Label[0])
                              
    return render_template('index.html', prediction_text='$ {}'.format(prediction), inr='₹ {}'.format(prediction * 73))


if __name__ == "__main__":
    app.run(host='localhost', port=8080)