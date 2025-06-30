from flask import Flask, render_template, request, send_file, jsonify
from keras.models import load_model
import os
import keras.preprocessing.image 
import numpy as np
from PIL import Image


app = Flask(__name__)

# Load the model
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the path to the model file relative to the current directory
model_filename = 'mobilenet_pneumonia_detection.h5'
model_path = os.path.join(current_directory, 'templates', model_filename)

# Load the model
model = load_model(model_path)
# Define a function to process the image and make predictions

def predict_pneumonia(img_path):
    try:
        img = Image.open(img_path)
        img_rgb = img.convert("RGB").resize((224, 224))  # Resize and ensure RGB mode
        img_tensor = keras.preprocessing.image.img_to_array(img_rgb) / 255.0  # Normalize
        img_tensor = np.expand_dims(img_tensor, axis=0)  # Add batch dimension
        
        # Predict
        prediction = model.predict(img_tensor)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        if predicted_class == 0:
            return {"class": "Normal", "confidence": f"{confidence:.2f}%"}
        else:
            return {"class": "Pneumonia", "confidence": f"{confidence:.2f}%"}
    except Exception as e:
        print(f"Error while reading the image: {e}")
        return {"error": "Image processing failed"}

@app.route('/')
def home():
    return render_template('page-1.html')


@app.route('/log-in')
def login():
    return render_template('log-in.html')

@app.route('/sign-in')
def signin():
    return render_template('sign-in.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/feature')
def feature():
    return render_template('feature.html')

@app.route('/aboutpg')
def aboutpg():
    return render_template('about-pg.html')

@app.route('/medical')
def medical():
    return render_template('medical-history.html')

@app.route('/diagnosis')
def diagnosis():
    return render_template('diagnosis.html')



@app.route('/final')
def final():
    return render_template('x-ray-1.html')

# Define a route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img = request.files['file']
        img_path = 'C:/Users/vishw/OneDrive/Documents/xray/AeroCheck - 1/AeroCheck - 1/static/public' + img.filename
        img.save(img_path)
        result = predict_pneumonia(img_path)
        return result
    prediction_result = {'result': 'some_prediction_result'}

    # Create a JSON response
    response = jsonify(prediction_result)

    # Add Cache-Control header to disable caching
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'

    return response


if __name__ == '__main__':
    app.run(debug=True)