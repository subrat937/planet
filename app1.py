from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import joblib
app = Flask(__name__)

model=joblib.load('model.job.lib')


@app.route('/')
def hello_world():
    return render_template("index.html")



@app.route('/predict',methods=['POST','GET'])
def predict():
    distance=request.form.get('dist')
    water=request.form.get('water')
    temperature=request.form.get('temp')
    mass=request.form.get('mass')
    radius=request.form.get('radius')
    output=model.predict([[distance,temperature,water,mass,radius]])
    
    if(output==[1]):
            return render_template("predict.html",pred='HABITABLE PLANET (means there is high chances of life on this planet) {}'.format(output))
    else:
           return render_template("predict.html",pred='NON-HABITABLE PLANET (means there is low chances of life on this planet) {}'.format(output))
if __name__ == '__main__':
    app.run(debug=True)