from flask import Flask, request, jsonify, render_template
import joblib
import os
import pandas as pd
import traceback

app = Flask(__name__)

# Load the model
MODEL_PATH = os.path.join('model', 'house_price_pipeline.pkl')
try:
    model_pipeline = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model_pipeline = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model_pipeline is None:
        return jsonify({'error': 'Machine learning model is not loaded. Please train the model first.'}), 500
        
    try:
        data = request.json
        
        # Extract features according to expected columns
        location = data.get('location', 'Other')
        area = float(data.get('area', 1000))
        bedrooms = int(data.get('bedrooms', 2))
        bathrooms = int(data.get('bathrooms', 2))
        balcony = int(data.get('balcony', 1))
        area_type = data.get('property_type', 'Super built-up  Area')
        
        # Create a DataFrame for the single prediction
        input_data = pd.DataFrame([{
            'location': location,
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'balcony': balcony,
            'area_type': area_type
        }])
        
        # Make prediction
        prediction = model_pipeline.predict(input_data)[0]
        
        return jsonify({
            'estimated_price': prediction,
            'formatted_price': f"₹{prediction:,.2f}"
        })
        
    except ValueError as ve:
        return jsonify({'error': f"Invalid input format: {str(ve)}"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
