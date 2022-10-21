from flask import Flask, render_template, request

import tensorflow as tf
from DocClf_Test_Preprocessing import *


app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():  # put application's code here
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    image_file = request.files['image_file']
    image_path = "./images/" + image_file.filename
    image_file.save(image_path)

    encoded_image = Preprocess(image_path).vector_encoder()
    encoded_image_reshaped = encoded_image.reshape(encoded_image.shape[1], 1)
    prep_image = np.expand_dims(encoded_image_reshaped, axis=0)
    doc_CNN_model = tf.keras.models.load_model("documents-CNN.model")
    #pipe = Pipeline([('preprocess', Preprocess(image_path)), ('cnn', CNN_model())])
    pred = (doc_CNN_model.predict([prep_image]) > 0.5)
    decoded_label = decoding_labels(pred)
    classification = decoded_label[0]
    print(classification)
    return render_template('index.html', prediction=classification)

if __name__ == '__main__':
    app.run()
