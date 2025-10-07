import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Force CPU, ignore GPUs


from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load Keras model
model = tf.keras.models.load_model("plant_disease_model.keras")
IMG_SIZE = 224

# Replace with your actual classes
class_names = [
    'Apple___Black_rot',
    'Tomato___Healthy',
    'Potato___Early_blight',
    # ... add all remaining class names here
]

# ✅ Home route to check API status
@app.route('/')
def home():
    return "✅ Plant Disease Detection API is running! Use POST /predict to send an image."

# ✅ Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    img_file = request.files['image']
    img = Image.open(img_file).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][class_idx])

    return jsonify({'class': class_names[class_idx], 'confidence': confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

