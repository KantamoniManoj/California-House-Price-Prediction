import json
import pickle
from flask import Flask,request,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Loading the model
regmodel=pickle.load(open('regressor.pkl','rb'))
## Loading the data
scaler=pickle.load(open('scaler.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])