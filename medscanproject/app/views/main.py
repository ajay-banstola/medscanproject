from flask import render_template, jsonify, Flask, redirect, url_for, request
from app import app
import random
import os
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


@app.route('/')
# disease_list = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', \
# 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', \
# 'Hernia']
@app.route('/upload')
def upload_file2():
    return render_template('index.html')


@app.route('/uploaded', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        new_model = load_model(r'C:\Users\Admin\Desktop\model.h5')
        # model= ResNet50(weights='imagenet')
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        p_good, p_ill = np.around(new_model.predict(x), decimals=2)[0]
        #preds = new_model.predict(x)
        #preds_decoded = decode_predictions(preds, top=3)[0]
        #print(decode_predictions(preds, top=3)[0])
        print(p_ill, p_good)
        f.save(path)
        return render_template('uploaded.html', title='Success', p_ill=p_ill, p_good=p_good, user_image=f.filename)


@app.route('/index')
def index():
    return render_template('index.html', title='Home')


@app.route('/map')
def map():
    return render_template('map.html', title='Map')


@app.route('/map/refresh', methods=['POST'])
def map_refresh():
    points = [(random.uniform(48.8434100, 48.8634100),
               random.uniform(2.3388000, 2.3588000))
              for _ in range(random.randint(2, 9))]
    return jsonify({'points': points})


@app.route('/contact')
def contact():
    return render_template('contact.html', title='Contact')
