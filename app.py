import os
import numpy as np
import tensorflow as tf
import joblib
from flask import Flask, request, jsonify

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="nutrisnack_model.tflite")
interpreter.allocate_tensors()

# Load scaler
scaler = joblib.load("scaler.pkl")

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def categorize_nutrient(value, very_low_threshold, low_range, moderate_range, high_range, very_high_min):
    """Categorize a nutrient value based on thresholds."""
    if value < very_low_threshold:
        return "Very Low"
    elif low_range[0] <= value <= low_range[1]:
        return "Low"
    elif moderate_range[0] <= value <= moderate_range[1]:
        return "Moderate"
    elif high_range[0] <= value <= high_range[1]:
        return "High"
    elif value > very_high_min:
        return "Very High"
    else:
        return "Unknown"

def predict_healthiness_tflite(fat, saturated_fat, carbohydrates, sugars, fiber, sodium, proteins):
    # Create input data
    input_data = np.array([[fat, saturated_fat, carbohydrates, sugars, fiber, sodium, proteins]])
    input_data_scaled = scaler.transform(input_data)  # Scale the input

    # Set input tensor to the scaled input data
    interpreter.set_tensor(input_details[0]['index'], input_data_scaled.astype(np.float32))

    # Run inference
    interpreter.invoke()

    # Get prediction
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # Determine healthiness
    health_status = "Healthy" if prediction > 0.5 else "Unhealthy"

    # Recommendation logic
    if health_status == "Healthy":
        if (6 <= fat < 10) and (3 <= saturated_fat < 5) and (25 <= carbohydrates < 30) and (3 <= sugars < 5) and (6 <= fiber < 10) and (14 <= proteins < 20) and (0.400 <= sodium < 0.600):
            recommendation = "Good to consume 1 time per day"
        elif (3 <= fat < 6) and (1.5 <= saturated_fat < 3) and (20 <= carbohydrates < 25) and (1.5 <= sugars < 3) and (3 <= fiber < 6) and (6 <= proteins < 14) and (0.200 <= sodium < 0.400):
            recommendation = "Good to consume 2 times per day"
        elif (0 <= fat < 3) and (0 <= saturated_fat < 1.5) and (0 <= carbohydrates < 20) and (0 <= sugars < 1.5) and (0 <= fiber < 3) and (0 <= proteins < 6) and (0 <= sodium < 0.200):
            recommendation = "Good to consume 3 times per day"
        else:
            recommendation = "Good to consume in moderate amounts"
    else:
        recommendation = "Better not to consume"

    # Categorize nutritional values
    categories = {
        "fat": categorize_nutrient(fat, 2, (2, 4), (4, 6), (6, 8), 8),
        "saturated_fat": categorize_nutrient(saturated_fat, 1, (1, 2), (2, 3), (3, 4), 4),
        "carbohydrates": categorize_nutrient(carbohydrates, 15, (15, 21), (21, 24), (24, 27), 27),
        "sugars": categorize_nutrient(sugars, 1, (1, 2), (2, 3), (3, 4), 4),
        "fiber": categorize_nutrient(fiber, 2, (2, 4), (4, 6), (6, 8), 8),
        "proteins": categorize_nutrient(proteins, 0, (0, 4), (4, 7), (7, 16), 16),
        "sodium": categorize_nutrient(sodium, 0.12, (0.12, 0.24), (0.24, 0.36), (0.36, 0.48), 0.48),
    }

    return {"health_status": health_status, "recommendation": recommendation, "categories": categories}

# Setup Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        nutritions = data.get('nutritions', {})
        fat = nutritions.get('fat', 0)
        saturated_fat = nutritions.get('saturated_fat', 0)
        carbohydrates = nutritions.get('carbohydrates', 0)
        sugars = nutritions.get('sugars', 0)
        fiber = nutritions.get('fiber', 0)
        sodium = nutritions.get('sodium', 0)
        proteins = nutritions.get('protein', 0)

        result = predict_healthiness_tflite(fat, saturated_fat, carbohydrates, sugars, fiber, sodium, proteins)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
