# House Price Prediction System

## Overview
The House Price Prediction System is a complete end-to-end machine learning web application. It predicts the estimated price of a property in India based on its features using regression models.

## Features
- End-to-end Machine Learning Pipeline
- Clean and modern HTML5/CSS3 frontend
- Flask REST API backend
- Model comparison and selection script
- Model serialization using Joblib

## Technologies
- **Backend:** Python 3, Flask
- **Machine Learning:** Pandas, NumPy, Scikit-learn, Joblib
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Visualization:** Matplotlib, Seaborn (in Notebook)

## Machine Learning
- **Preprocessing:** Handling missing values, converting string intervals to floats, removing outliers.
- **Feature Engineering:** One-hot encoding categorical variables, scaling numerical features.
- **Models:** Evaluates Linear Regression, Decision Tree Regressor, Random Forest Regressor, and Gradient Boosting Regressor.
- **Evaluation Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R-Squared (R²).
- **Final Model:** The script automatically selects and serializes the model with the best R² score.

## Project Structure
```
house-price-prediction/
│
├── dataset/
│   └── houses.csv
│
├── model/
│   └── house_price_pipeline.pkl
│
├── notebooks/
│   └── house_price_prediction.ipynb
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Training
To train the model and generate the pipeline file:
```bash
python train_model.py
```
This loads the dataset, cleans it, trains four models, outputs their evaluation metrics, and saves the best model to `model/house_price_pipeline.pkl`.

## Running the application
Start the Flask backend:
```bash
python app.py
```
Open a web browser and navigate to `http://127.0.0.1:5000`

## API
**Endpoint:** `POST /predict`

**Example Request:**
```json
{
  "location": "Whitefield",
  "area": 1500,
  "bedrooms": 3,
  "bathrooms": 2,
  "balcony": 1,
  "property_type": "Super built-up  Area"
}
```

**Example Response:**
```json
{
  "estimated_price": 8500000.0,
  "formatted_price": "₹8,500,000.00"
}
```

## Disclaimer
The result provided by this application is an ML-based estimate using historical data and should not be treated as an official professional property valuation.
