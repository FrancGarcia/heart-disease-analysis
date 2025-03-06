from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS  # Import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained ML model
# model = joblib.load('heart_disease_model.pkl')

@app.route("/")
def home():
    return "Heart Disease Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from the frontend (input fields)
        data = request.json
        
        # For debugging, print the received data
        print("Received data:", data)

        # Uncomment and replace with actual feature extraction once model is loaded
        # features = np.array([
        #     data["age"], 
        #     data["weight"], 
        #     data["cholesterol"], 
        #     data["blood_pressure"], 
        #     data["sugar"], 
        #     data["family_history"]
        # ]).reshape(1, -1)

        # Make a prediction (replace this with model prediction later)
        # prediction = model.predict(features)[0]
        prediction = "Mock Prediction: No Heart Disease"  # Temporary mock result

        # Return the prediction to the frontend
        result = {
            "prediction": prediction
        }
        return jsonify(result)

    except Exception as e:
        # Handle any error that might occur
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
