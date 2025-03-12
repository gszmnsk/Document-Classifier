from flask import Flask, render_template, request
import pickle
from PIL import Image
import os
from preprocessing.preprocessing_docs import *

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():  # put application's code here
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    image_file = request.files['image_file']
    UPLOAD_FOLDER =  "./static/"
    image_path = UPLOAD_FOLDER + image_file.filename
    if image_file:
        # Check if the file is a TIFF image
        if image_file.filename.lower().endswith(".tif") or image_file.filename.lower().endswith(".tiff"):
            # Open TIFF and convert to PNG
            img = Image.open(image_file)
            image_path = os.path.join(UPLOAD_FOLDER, image_file.filename.replace(".tif", ".png").replace(".tiff", ".png"))
            img.convert("RGB").save(image_path, "PNG")
        else:
            image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_path)
    # SVM
    encoded_image = Preprocess(image_path).vector_encoder()
    svm_model = pickle.load(open('../models/saved_models/SVM_model.sav', 'rb'))
    predicted = svm_model.predict(encoded_image)
    proba = svm_model.predict_proba(encoded_image)
    classification = None
    ###
    # CNN
    # encoded_image_reshaped = encoded_image.reshape(encoded_image.shape[1], 1)
    # prep_image = np.expand_dims(encoded_image_reshaped, axis=0)
    # doc_CNN_model = tf.keras.models.load_model("documents-CNN.model")
    # pipe = Pipeline([('preprocess', Preprocess(image_path)), ('cnn', CNN_model())])
    # pred = (doc_CNN_model.predict([prep_image]) > 0.5)
    # decoded_label = decoding_labels(pred)
    # classification = decoded_label[0]
    ###
    if predicted[0] == 0:
        classification = "News"
        prob = round(proba[0][0] * 100, 2)
    elif predicted[0] == 1:
        classification = "Resume"
        prob = round(proba[0][1] * 100, 2)
    elif predicted[0] == 2:
        classification = "Scientific"
        prob = round(proba[0][2] * 100, 2)
    print(prob)
    return render_template('index.html', prediction=classification, image_path=image_path, probability=prob)

if __name__ == '__main__':
    app.run()
