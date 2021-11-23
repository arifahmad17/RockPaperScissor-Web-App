#app.py
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from keras.preprocessing import image
import numpy as np
 
from keras.models import load_model
app = Flask(__name__)

dict={0:'Paper',1:'Rock',2:'Scissors'}

model = load_model('model.h5')
model.make_predict_function()
def predict_label(img_path):
    dict={0:'Paper',1:'Rock',2:'Scissors'}
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images,batch_size=10)
    cl=np.argmax(classes)
    return(dict[cl])
    


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html",prediction=p ,img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug=True)
        
