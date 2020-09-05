# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 15:20:09 2020

@author: Ranna
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/results',methods=['POST'])
def results():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

@app.route('/predict',methods=['POST'])
def predict():
      
    int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    prediction = model.predict(int_features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Revolving credit balance should be $ {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)







