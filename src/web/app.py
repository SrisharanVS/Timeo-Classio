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
import logging 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.info("Received /predict request")

        # Get model selector
        selector = get_model_selector()
        logger.info("ModelSelector loaded")

        # Read the uploaded file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        logger.info("CSV file read successfully")

        # Check for required column
        if 'point_value' not in df.columns:
            logger.warning("Missing 'point_value' column in uploaded CSV")
            raise HTTPException(status_code=400, detail="CSV file must have a 'point_value' column")

        # Handle missing values
        if df['point_value'].isna().any():
            logger.info(f"Detected {df['point_value'].isna().sum()} missing values in 'point_value' column")
            df['point_value'] = df['point_value'].fillna(method='ffill').fillna(method='bfill')
            if df['point_value'].isna().any():
                df['point_value'] = df['point_value'].interpolate(method='linear', limit_direction='both')
            if df['point_value'].isna().any():
                mean_value = df['point_value'].mean()
                if pd.isna(mean_value):
                    mean_value = 0
                df['point_value'] = df['point_value'].fillna(mean_value)
            logger.info("Missing values handled")

        data = df['point_value'].to_numpy()
        has_timestamps = 'point_timestamp' in df.columns
        timestamps = df['point_timestamp'].values if has_timestamps else None

        train_size = int(len(data) * (1 - test_size))
        train_data = data[:train_size]
        test_data = data[train_size:]
        train_timestamps = timestamps[:train_size] if timestamps is not None else None
        test_timestamps = timestamps[train_size:] if timestamps is not None else None
        logger.info(f"Data split into train ({len(train_data)}) and test ({len(test_data)})")

        best_model_name, predicted_mape = selector.select_model(train_data)
        logger.info(f"Selected best model: {best_model_name} with predicted MAPE: {predicted_mape}")
        model = selector.get_model(best_model_name)

        if model is None:
            logger.error(f"Model {best_model_name} could not be loaded")
            raise HTTPException(status_code=500, detail=f"Could not load model {best_model_name}")

        model.fit(train_data)
        logger.info(f"Model {best_model_name} trained")

        test_predictions = model.predict(len(test_data))
        logger.info("Test predictions generated")

        test_mape = calculate_mape(test_data, test_predictions)
        logger.info(f"Test MAPE: {test_mape}")

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

        logger.info("Formatted train and test data for response")

        return {
            "best_model": best_model_name,
            "test_mape": float(test_mape) if test_mape is not None else None,
            "train_data": train_data_formatted,
            "test_data": test_data_formatted
        }

    except Exception as e:
        logger.exception("Exception during prediction")
        raise HTTPException(status_code=500, detail=str(e))

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
