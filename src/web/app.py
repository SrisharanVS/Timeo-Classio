from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
import io
from typing import List, Dict
import sys
import os

# Add the project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.classifier.model_selector import ModelSelector
from src.features.extractor import TimeSeriesFeatureExtractor

# Get the absolute path to the web directory and models directory
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

app = FastAPI(title="Time Series Forecasting API")

# Mount static files and templates
app.mount("/static", StaticFiles(directory=os.path.join(WEB_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(WEB_DIR, "templates"))

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model selector as None - will be loaded on demand
model_selector = None

def get_model_selector():
    """Get or initialize the model selector"""
    global model_selector
    if model_selector is None:
        try:
            model_selector = ModelSelector(models_dir=MODELS_DIR)
            model_selector.load()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Models not available. Please run training first. Error: {str(e)}"
            )
    return model_selector

def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate MAPE with proper handling of edge cases."""
    # Convert inputs to numpy arrays
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    
    # Check for length mismatch
    if len(actual) != len(predicted):
        return float('inf')
    
    # Handle empty arrays
    if len(actual) == 0:
        return float('inf')
    
    # Check for NaN or inf values
    if np.any(np.isnan(actual)) or np.any(np.isnan(predicted)) or \
       np.any(np.isinf(actual)) or np.any(np.isinf(predicted)):
        # Try to clean the data
        try:
            # Replace NaN and inf values
            actual = pd.Series(actual).fillna(method='ffill').fillna(method='bfill').fillna(0).values
            predicted = pd.Series(predicted).fillna(method='ffill').fillna(method='bfill').fillna(0).values
        except:
            return float('inf')
    
    # Filter out points where actual is zero or close to zero to avoid division by zero
    epsilon = 1e-10
    mask = np.abs(actual) > epsilon
    
    if not np.any(mask):
        return float('inf')  # All values are zero or close to zero
    
    actual_filtered = actual[mask]
    predicted_filtered = predicted[mask]
    
    try:
        # Calculate MAPE using sklearn
        mape = mean_absolute_percentage_error(actual_filtered, predicted_filtered) * 100
        
        # Cap extremely large values
        return min(mape, 1000.0)  # Cap at 1000%
    except Exception as e:
        print(f"Error calculating MAPE: {str(e)}")
        return float('inf')

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...), test_size: float = 0.2) -> Dict:
    """
    Upload a time series CSV file and get predictions for test set.
    
    The CSV file should have a 'point_value' column, and optionally a 'point_timestamp' column.
    """
    try:
        # Get model selector (will raise HTTPException if models aren't available)
        selector = get_model_selector()
        
        # Read the uploaded file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Extract the point_value column and convert to numpy array
        if 'point_value' not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="CSV file must have a 'point_value' column"
            )
        
        # Handle missing values in the data
        # First, check if there are any missing values
        if df['point_value'].isna().any():
            print(f"Input data contains {df['point_value'].isna().sum()} missing values out of {len(df)} points")
            
            # 1. Forward fill followed by backward fill (handles internal NaNs)
            df['point_value'] = df['point_value'].fillna(method='ffill').fillna(method='bfill')
            
            # 2. If still have NaNs (e.g., at the beginning/end), use interpolation
            if df['point_value'].isna().any():
                df['point_value'] = df['point_value'].interpolate(method='linear', limit_direction='both')
            
            # 3. Last resort: fill any remaining NaNs with the mean
            if df['point_value'].isna().any():
                mean_value = df['point_value'].mean()
                if pd.isna(mean_value):  # If mean is also NaN (all values are NaN)
                    mean_value = 0  # Use 0 as a fallback
                df['point_value'] = df['point_value'].fillna(mean_value)
        
        data = df['point_value'].to_numpy()
        
        # Check for timestamps
        has_timestamps = 'point_timestamp' in df.columns
        timestamps = df['point_timestamp'].values if has_timestamps else None
        
        # Split data into train and test sets
        train_size = int(len(data) * (1 - test_size))
        train_data = data[:train_size]
        test_data = data[train_size:]
        train_timestamps = timestamps[:train_size] if timestamps is not None else None
        test_timestamps = timestamps[train_size:] if timestamps is not None else None
        
        # Get model selection and predictions using training data
        best_model_name, predicted_mape = selector.select_model(train_data)
        model = selector.get_model(best_model_name)
        
        if model is None:
            raise HTTPException(
                status_code=500,
                detail=f"Could not load model {best_model_name}"
            )
        
        # Train on training data and predict test data
        model.fit(train_data)
        test_predictions = model.predict(len(test_data))
        
        # Calculate MAPE on test data
        test_mape = calculate_mape(test_data, test_predictions)
        
        # Format data for visualization
        train_data_formatted = [
            {
                "timestamp": str(ts) if has_timestamps else str(i),
                "value": float(val) if not np.isnan(val) else None
            }
            for i, (ts, val) in enumerate(zip(train_timestamps if has_timestamps else range(train_size), train_data))
        ]
        
        test_data_formatted = [
            {
                "timestamp": str(ts) if has_timestamps else str(i + train_size),
                "value": float(val) if not np.isnan(val) else None,
                "prediction": float(pred) if not np.isnan(pred) else None
            }
            for i, (ts, val, pred) in enumerate(zip(
                test_timestamps if has_timestamps else range(train_size, len(data)),
                test_data,
                test_predictions
            ))
        ]
        
        return {
            "best_model": best_model_name,
            "test_mape": float(test_mape) if test_mape is not None else None,
            "train_data": train_data_formatted,
            "test_data": test_data_formatted
        }
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Try to load models to check if they're available
        selector = get_model_selector()
        
        # Check if forecasting models are loaded
        if not hasattr(selector, 'forecasting_models') or not selector.forecasting_models:
            return {
                "status": "degraded", 
                "models_loaded": False, 
                "error": "Forecasting models not loaded properly."
            }
            
        return {"status": "healthy", "models_loaded": True}
    except HTTPException as e:
        return {"status": "degraded", "models_loaded": False, "error": e.detail}
    except Exception as e:
        return {"status": "error", "models_loaded": False, "error": str(e)} 