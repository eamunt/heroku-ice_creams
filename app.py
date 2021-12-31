from flask import Flask,render_template,url_for,request
from flask_material import Material

import pandas as pd
import numpy as np

import joblib

app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        female = request.form['female']
        if female == "0":
            gender = "Nam"
        elif female == "1":
            gender = "Nữ"
        video = request.form['video']
        puzzle = request.form['puzzle']
        model = request.form['model']
        sample_data = [female, video, puzzle]
        clean_data = [float(i) for i in sample_data]
        ex1 = np.array(clean_data).reshape(1,-1)
        if model == "tree":
            ic_model = joblib.load("data/dt_model_icecream.pkl")
        elif model == "bayes":
            ic_model = joblib.load("data/nb_model_icecream.pkl")
        elif model == "knn":
            ic_model = joblib.load("data/knn_model_icecream.pkl")
        result_prediction = ic_model.predict(ex1)
        if result_prediction == [1]:
            result_prediction = "Vanilla"
        elif result_prediction == [2]:
            result_prediction = "Chocolate"
        elif result_prediction == [3]:
            result_prediction = "Strawberry"
    return render_template("index.html", gender=gender,
        video=video,
        puzzle=puzzle,
        clean_data=clean_data,
        result_prediction=result_prediction)

if __name__ == '__main__':
    app.run(debug=True)