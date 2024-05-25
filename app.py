import pandas as pd
import numpy as np
from flask import Flask,render_template,request,jsonify 
import pickle

app = Flask(__name__)

model = pickle.load(open("reg_model.pkl","rb"))
scaling = pickle.load(open("stander_scalar.pkl","rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api",methods=["POST"])
def predict_api():
    data = request.json['data']
    print(data)
    data.values()
    print(np.array(list(data.values())).reshape(1,-1))
    scaled_data = scaling.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(scaled_data) 
    print(output[0])
    return jsonify(output[0])

@app.route("/predict",methods=["POST"])
def predict():
    
    data = [float(i) for i in request.form.values()]
    # print(data)
    # data.values()
    final_output = np.array(data).reshape(1,-1)
    scaled_data = scaling.transform(final_output)
    output = model.predict(scaled_data) 
    if output > 0 : 
        return render_template('home.html' ,prediction_text = "This is your house price in lacks based on your input is {}".format(round(output[0],2)))
    else:
        return render_template('home.html' ,prediction_text = "There is some invalid entry please recheck, Thanks for your time")


if __name__ == "__main__":
    app.run(debug=True)


