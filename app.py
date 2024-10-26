import os
import json
from PIL import Image
import base64

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from pymongo import MongoClient

app = Flask(__name__)

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "plant_disease_prediction_model.h5")
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(os.path.join(working_dir, "class_indices.json")))

# Configure MongoDB
client = MongoClient('mongodb://mongodb-service:27017/')
db = client['deep']
collection = db['predictions']

# Configure upload folder
upload_dir = os.path.join(working_dir, 'uploads')
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)
app.config['UPLOAD_FOLDER'] = upload_dir


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(file, target_size=(224, 224)):
    # Load the image
    img = Image.open(file)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, file, class_indices):
    preprocessed_img = load_and_preprocess_image(file)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Function to convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/home')
def homepage():
    return render_template('home.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Get prediction
            prediction = predict_image_class(model, file_path, class_indices)
            # Save prediction and image to MongoDB
            image_base64 = image_to_base64(file_path)
            data = {'prediction': prediction, 'image_base64': image_base64}
            collection.insert_one(data)
            # Render upload page with uploaded image
            return render_template('upload.html', image_file=filename)
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict():
    filename = request.form['filename']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    prediction = predict_image_class(model, file_path, class_indices)
    return render_template('result.html', prediction=prediction, image_file=filename)


@app.route('/last_prediction')
def last_prediction():
    # Retrieve the last prediction from MongoDB
    last_data = collection.find_one(sort=[('_id', -1)])
    if last_data:
        last_prediction = last_data['prediction']
        last_image_base64 = last_data['image_base64']
        return render_template('last_prediction.html', last_prediction=last_prediction, last_image_base64=last_image_base64)
    else:
        return 'No predictions available'


@app.route('/all_predictions')
def all_predictions():
    # Retrieve all predictions from MongoDB
    all_data = collection.find()
    predictions = []
    for data in all_data:
        predictions.append({'prediction': data['prediction'], 'image_base64': data['image_base64']})
    return render_template('all_predictions.html', predictions=predictions)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
