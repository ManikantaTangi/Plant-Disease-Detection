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

        # Get top-3 predictions
        top_indices = predictions[0].argsort()[-3:][::-1]
        top_classes = [class_names[i] for i in top_indices]
        top_confidences = [float(predictions[0][i]) for i in top_indices]

        # Combine into list of dicts
        top_results = [
            {'class': top_classes[i], 'confidence': top_confidences[i]}
            for i in range(3)
        ]

        # Best prediction (for backward compatibility)
        best_idx = int(top_indices[0])

        return jsonify({
            'top_3_predictions': top_results,
            'best_prediction': {
                'class': class_names[best_idx],
                'confidence': float(predictions[0][best_idx])
            }
        }), 200

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500
