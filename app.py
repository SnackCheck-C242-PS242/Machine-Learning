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
        if (24 <= sugars < 50) and (1 <= sodium < 2) and (34 <= fat < 67):
            recommendation = "Good to consume 1 time per day"
        elif (15 <= sugars < 24) and (0.7 <= sodium < 1) and (22 <= fat < 34):
            recommendation = "Good to consume 2 times per day"
        elif (0 <= sugars < 15) and (0 <= sodium < 0.6) and (0 <= fat < 22):
            recommendation = "Good to consume 3 times per day"
        else:
            recommendation = "Good to consume in moderate amounts"
    else:
        recommendation = "Better not to consume"

    return {"health_status": health_status, "recommendation": recommendation}

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
