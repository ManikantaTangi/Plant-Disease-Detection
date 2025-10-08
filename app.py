import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

# Constants
IMG_SIZE = 224
MODEL_PATH = "plant_disease_model.keras"

# Replace with your actual trained class names
class_names = [
    'Apple___Black_rot',
    'Tomato___Healthy',
    'Potato___Early_blight'
    # Add all remaining class names here (no "..." placeholder)
]

# Lazy-loaded model
model = None


def load_model():
    """Load the TensorFlow model once."""
    global model
    if model is None:
        print("🔄 Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
    return model


@app.route('/')
def home():
    return jsonify({'message': '✅ Plant Disease Detection API is running!'}), 200


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure image is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        img_file = request.files['image']

        # Validate and preprocess image
        try:
            img = Image.open(img_file).convert('RGB')
        except Exception as e:
            return jsonify({'error': f'Invalid image file: {str(e)}'}), 400

        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Load model (if not already loaded)
        model_instance = load_model()

        # Run prediction
        predictions = model_instance.predict(img_array)
        if predictions is None or len(predictions) == 0:
            return jsonify({'error': 'Model returned no predictions'}), 500

        class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][class_idx])

        return jsonify({
            'class': class_names[class_idx],
            'confidence': confidence
        }), 200

    except Exception as e:
        # Catch any unhandled errors so Flask doesn't crash (→ 502)
        print("❌ Error during prediction:", str(e))
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health():
    """Simple health check for Render."""
    return jsonify({'status': 'ok'}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
