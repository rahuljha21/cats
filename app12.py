from flask import Flask,render_template
import requests
import numpy as np

import keras as keras
from keras.preprocessing.image import img_to_array,load_img
from keras.models import load_model

model=load_model('app.h5')
def catpredict(img):
    test_img=load_img(img,target=(150,150))
    test_img=img_to_array(test_img)/255.0
    test_img=np.expand(test_img,axis=0)
    result=model.predict(test_img).round(3)
    pred=np.argmax(result)
    if(pred==0):
        return 'cat'
    else:
        return 'dog'
app=Flask(__name__)
@app.route('/predict',methods=['Get','Post'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
         
        
        
 
        print("@@ Predicting class......")
        pred, output_page =catpredict(filename)
        return pred
if __name__=='__main__':
    app.run(debug=True)