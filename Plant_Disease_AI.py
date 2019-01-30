import flask
import numpy as np
import pickle
import cv2
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import os
from flask import Flask, flash, request, redirect, url_for,send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

labels=pickle.load(open('label_transform.pkl', "rb"))
model=load_model('modelpath')

app = flask.Flask(__name__,template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
    if flask.request.method == 'POST':
        data = {"success": "Error"}
        file = request.files['file']
        
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            fName=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image = cv2.imread(fName)
            if image is not None :
                image = cv2.resize(image, (128,128))
                tstImg=img_to_array(image)
            
            image = np.expand_dims(tstImg, axis=0)
            np_image = np.array(image, dtype=np.float16)
			
			predict=model.predict(np_image)
			prediction=model.predict_classes(np_image)
			data["Class"]=labels.classes_[prediction][0].replace('_',' ')
			data["predict"]='{0:.2f}'.format(predict.max()*100)
			data["success"]= "Success"
			data["img"]=filename
            
            return flask.render_template('index.html', scroll='frm',result=data)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug = True)