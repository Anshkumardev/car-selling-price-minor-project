from flask import Flask,request,jsonify
import pickle
import numpy as np
import pandas as pd
import sklearn

pipe = pickle.load(open('SellCarPriceModel.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello world"

@app.route('/predict',methods=['POST'])
def predict():
    name = request.form.get('name')
    company = request.form.get('company')
    year = request.form.get('year')
    kms_driven = request.form.get('kms_driven')
    fuel_type = request.form.get('fuel_type')

    result = pipe.predict(pd.DataFrame([[name, company, year, kms_driven, fuel_type]],columns=["name", "company", "year", "kms_driven", "fuel_type"]))


    return jsonify({'predicted_price':str(result[0])})


if __name__ == "__main__":
    app.run(debug=True)

