import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def convert_sqft_to_num(x):
    if pd.isnull(x):
        return np.nan
    tokens = str(x).split('-')
    if len(tokens) == 2:
        try:
            return (float(tokens[0]) + float(tokens[1])) / 2
        except:
            return np.nan
    try:
        return float(x)
    except:
        return np.nan

def clean_dataset(df):
    print("Initial dataset shape:", df.shape)
    
    # Drop irrelevant columns
    df = df.drop(['society', 'availability'], axis=1, errors='ignore')
    
    # Handle missing values
    df = df.dropna(subset=['location', 'size', 'bath', 'total_sqft'])
    
    # Extract integer from size (e.g. '2 BHK' -> 2)
    df['bedrooms'] = df['size'].apply(lambda x: int(x.split(' ')[0]) if pd.notnull(x) else np.nan)
    df = df.drop('size', axis=1)
    
    # Convert total_sqft to numeric
    df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
    df = df.dropna(subset=['total_sqft'])
    
    # Simplify location by grouping infrequent locations to 'Other'
    location_stats = df['location'].value_counts()
    location_less_than_10 = location_stats[location_stats <= 10]
    df['location'] = df['location'].apply(lambda x: 'Other' if x in location_less_than_10 else x)
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Rename features to be cleaner
    df = df.rename(columns={
        'total_sqft': 'area',
        'bath': 'bathrooms'
    })
    
    # Remove obvious outliers (e.g., bedrooms > 20 or area < 100)
    df = df[(df['bedrooms'] < 20) & (df['area'] >= 100)]
    
    # Target price is in Lakhs, converting to total Rupees
    df['price'] = df['price'] * 100000
    
    print("Cleaned dataset shape:", df.shape)
    return df

def main():
    dataset_path = os.path.join('dataset', 'houses.csv')
    if not os.path.exists(dataset_path):
        print("Dataset not found at", dataset_path)
        return
        
    df = pd.read_csv(dataset_path)
    
    print("--- Inspecting Dataset ---")
    print(df.info())
    print(df.head())
    
    df = clean_dataset(df)
    
    # Features and Target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Identify numerical and categorical columns
    num_cols = ['area', 'bedrooms', 'bathrooms', 'balcony']
    cat_cols = ['area_type', 'location']
    
    print("\nNumerical columns:", num_cols)
    print("Categorical columns:", cat_cols)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create preprocessing pipelines
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    # Models to evaluate
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=50, random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
    }
    
    best_model_name = None
    best_r2 = -float('inf')
    best_pipeline = None
    
    print("\n## Model Performance\n")
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(name)
        print(f"MAE:  Rs. {mae:,.2f}")
        print(f"RMSE: Rs. {rmse:,.2f}")
        print(f"R2:   {r2:.4f}\n")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline
            
    print(f"Best Model: {best_model_name} (R2: {best_r2:.4f})")
    
    # Save best model
    os.makedirs('model', exist_ok=True)
    model_path = os.path.join('model', 'house_price_pipeline.pkl')
    joblib.dump(best_pipeline, model_path)
    print(f"\nModel pipeline saved to {model_path}")

if __name__ == '__main__':
    main()
