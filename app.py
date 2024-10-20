from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = 'C:/Users/ishit/OneDrive/Desktop/flipkart/Fresh_Rotten_Fruits_MobileNetV2_Transfer_Learning.h5'
model = load_model(MODEL_PATH)

# Define the target image size
TARGET_SIZE = (224, 224)

# Class labels
CLASS_LABELS = ['Fresh Apple', 'Fresh Banana', 'Fresh Orange', 'Rotten Apple', 'Rotten Bananas', 'Rotten Orange']

# Define the directory where images are saved
IMAGE_DIR = 'C:/Users/ishit/OneDrive/Desktop/flipkart/dataset/dataset/test'

# Define shelf life in days for each class
SHELF_LIFE = {
    'Fresh Apple': 10,
    'Fresh Banana': 5,
    'Fresh Orange': 15,
    'Rotten Apple': 0,
    'Rotten Bananas': 0,
    'Rotten Orange': 0
}

# Home route to display the form for image upload
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save the uploaded file to the test folder (no need to move it to static/uploads)
        file_path = os.path.join(IMAGE_DIR, file.filename)
        file.save(file_path)
        
        # Preprocess the image
        img = load_img(file_path, target_size=TARGET_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Perform prediction
        prediction = model.predict(img_array)
        predicted_class = CLASS_LABELS[np.argmax(prediction)]
        
        # Get the shelf life for the predicted class
        shelf_life = SHELF_LIFE[predicted_class]
        
        return render_template('index.html', prediction=predicted_class, shelf_life=shelf_life, image_path=file.filename)

# Route to serve images from the dataset/dataset/test directory
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(IMAGE_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True)
