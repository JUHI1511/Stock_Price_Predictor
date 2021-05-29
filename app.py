# -*- coding: utf-8 -*-
"""
Created on Sun May 16 12:51:48 2021

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 15 21:16:10 2021

@author: hp
"""

from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

# load the model from disk
filename = 'nlp_model.pkl'
rc = pickle.load(open(filename, 'rb'))
vectorizer_x=pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = vectorizer_x.transform(data).toarray()
		my_prediction = rc.predict(vect)
	return render_template('index.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True,use_reloader=False)