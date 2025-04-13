import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import torch
from typing import Tuple
import torch.nn as nn
from sklearn.metrics import mean_absolute_percentage_error

from src.features.extractor import TimeSeriesFeatureExtractor
from src.forecasting.models import ModelFactory, ARIMAModel, ExponentialSmoothingModel, SimpleMovingAverageModel, ProphetModel
from src.classifier.models import MAPEPredictor

class ModelSelector:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.feature_extractor = TimeSeriesFeatureExtractor()
        self.feature_scaler = None
        self.mape_scaler = None
        self.nn_model = None
        self.feature_cols = None
        self.mape_cols = None
        self.forecasting_models = {}
        self.load_models()
        
    def load_models(self):
        """Load all pre-trained models and scalers"""
        # Load feature scaler
        scaler_path = os.path.join(self.models_dir, "feature_scaler.joblib")
        if not os.path.exists(scaler_path):
            raise ValueError("Feature scaler not found. Please run training first.")
        self.feature_scaler = joblib.load(scaler_path)
        
        # Load MAPE scaler
        mape_scaler_path = os.path.join(self.models_dir, "mape_scaler.joblib")
        if not os.path.exists(mape_scaler_path):
            raise ValueError("MAPE scaler not found. Please run training first.")
        self.mape_scaler = joblib.load(mape_scaler_path)
        
        # Load neural network
        nn_path = os.path.join(self.models_dir, "mape_predictor.pth")
        if not os.path.exists(nn_path):
            raise ValueError("Neural network model not found. Please run training first.")
            
        # Load model data
        model_data = torch.load(nn_path)
        self.feature_cols = model_data['feature_cols']
        self.mape_cols = model_data['mape_cols']
        
        # Initialize and load neural network
        input_size = len(self.feature_cols)
        self.nn_model = MAPEPredictor(input_size=input_size)
        self.nn_model.load_state_dict(model_data['state_dict'])
        self.nn_model.eval()
        
        # Load pre-trained forecasting models
        for model_name in [m.name.lower() for m in ModelFactory.get_all_models()]:
            model_path = os.path.join(self.models_dir, f"{model_name}_model.joblib")
            if os.path.exists(model_path):
                self.forecasting_models[model_name] = joblib.load(model_path)
    
    def prepare_training_data(self, time_series_data: list, test_size: float = 0.2):
        """
        Prepare training data by extracting features and finding best models
        
        Args:
            time_series_data: List of time series arrays
            test_size: Proportion of data to use for testing
        """
        features_list = []
        best_models = []
        
        for ts in time_series_data:
            # Extract features
            features = self.feature_extractor.extract_features(ts)
            features_list.append(features)
            
            # Find best model
            best_model = self._find_best_model(ts)
            best_models.append(best_model)
        
        # Combine features
        X = pd.concat(features_list, ignore_index=True)
        y = self.label_encoder.transform(best_models)
        
        # Split data
        return train_test_split(X, y, test_size=test_size, random_state=42)
    
    def _find_best_model(self, time_series: np.ndarray) -> str:
        """
        Find the best model for a given time series by comparing MAPE values
        """
        # Split data into train and test
        train_size = int(len(time_series) * 0.8)
        train_data = time_series[:train_size]
        test_data = time_series[train_size:]
        
        best_mape = float('inf')
        best_model_name = None
        
        # Get fresh instances of models
        models = [
            ARIMAModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)),
            ExponentialSmoothingModel(),
            SimpleMovingAverageModel(),
            ProphetModel()
        ]
        
        for model in models:
            try:
                # Fit model
                model.fit(train_data)
                
                # Generate predictions
                predictions = model.predict(len(test_data))
                
                # Calculate MAPE using sklearn
                # Filter out zeros from test_data to avoid division by zero
                mask = np.abs(test_data) > 1e-10
                if np.any(mask):
                    test_filtered = test_data[mask]
                    pred_filtered = predictions[mask]
                    
                    if len(test_filtered) > 0:
                        mape = mean_absolute_percentage_error(test_filtered, pred_filtered) * 100
                    else:
                        mape = float('inf')
                else:
                    mape = float('inf')
                
                # Update best model if necessary
                if mape < best_mape:
                    best_mape = mape
                    best_model_name = model.name
                    
            except Exception as e:
                print(f"Error with model {model.name}: {str(e)}")
                continue
        
        return best_model_name
    
    def train(self, time_series_data: list):
        """
        Train the model selector using a list of time series data
        """
        X_train, X_test, y_train, y_test = self.prepare_training_data(time_series_data)
        self.classifier.fit(X_train, y_train)
        
        # Calculate and print accuracy
        train_accuracy = self.classifier.score(X_train, y_train)
        test_accuracy = self.classifier.score(X_test, y_test)
        print(f"Training accuracy: {train_accuracy:.2f}")
        print(f"Testing accuracy: {test_accuracy:.2f}")
    
    def select_model(self, time_series: np.ndarray) -> Tuple[str, float]:
        """
        Select the best model for a given time series
        Returns:
            tuple: (best_model_name, predicted_mape)
        """
        # Preprocess time series to handle NaN values
        time_series = np.asarray(time_series)
        if np.any(np.isnan(time_series)):
            print(f"Input time series contains {np.sum(np.isnan(time_series))} NaN values")
            
            # Clean the time series
            ts_series = pd.Series(time_series)
            # Forward fill then backward fill
            ts_cleaned = ts_series.fillna(method='ffill').fillna(method='bfill')
            
            # If still have NaNs, use interpolation
            if ts_cleaned.isna().sum() > 0:
                ts_cleaned = ts_cleaned.interpolate(method='linear', limit_direction='both')
            
            # Last resort: fill with mean or zero
            if ts_cleaned.isna().sum() > 0:
                mean_val = ts_cleaned.mean()
                if pd.isna(mean_val):
                    mean_val = 0
                ts_cleaned = ts_cleaned.fillna(mean_val)
            
            # Convert back to numpy array
            time_series = ts_cleaned.values
        
        # Extract features
        features = self.feature_extractor.extract_features(time_series)
        
        # Ensure features are in the correct order
        feature_values = []
        for col in self.feature_cols:
            if col in features.columns:
                feature_values.append(features.iloc[0][col])
            else:
                raise ValueError(f"Feature {col} not found in extracted features")
        
        # Handle NaN in features
        feature_array = np.array(feature_values).reshape(1, -1)
        if np.any(np.isnan(feature_array)):
            print(f"Features contain {np.sum(np.isnan(feature_array))} NaN values")
            feature_df = pd.DataFrame(feature_array, columns=self.feature_cols)
            
            # First try: Replace NaN with column mean
            column_means = feature_df.mean()
            for col in feature_df.columns:
                if pd.isna(column_means[col]):
                    column_means[col] = 0  # Use 0 if mean is also NaN
            
            feature_df = feature_df.fillna(column_means)
            
            # Update feature_array
            feature_array = feature_df.values
        
        # Scale features
        scaled_features = self.feature_scaler.transform(feature_array)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(scaled_features)
        
        # Predict MAPE values
        with torch.no_grad():
            scaled_mapes = self.nn_model(features_tensor)
            
        # Convert back to original scale
        predicted_mapes = self.mape_scaler.inverse_transform(scaled_mapes.numpy())
        
        # Get best model
        best_model_idx = np.argmin(predicted_mapes[0])
        best_model_name = self.mape_cols[best_model_idx].replace('MAPE_', '').lower()
        predicted_mape = predicted_mapes[0][best_model_idx]
        
        # Initialize the best model with the time series data
        model = self.get_model(best_model_name)
        model.fit(time_series)
        
        return best_model_name, predicted_mape
    
    def get_model(self, model_name: str):
        """Get a pre-trained forecasting model by name"""
        model_name = model_name.lower()
        if model_name not in self.forecasting_models:
            # If model not loaded, create a new one
            model = ModelFactory.get_model(model_name)
            self.forecasting_models[model_name] = model
        return self.forecasting_models[model_name]
    
    def save(self, path: str):
        """Save the model selector and all its components"""
        os.makedirs(path, exist_ok=True)
        
        # Save feature scaler
        joblib.dump(self.feature_scaler, os.path.join(path, "feature_scaler.joblib"))
        
        # Save MAPE scaler
        joblib.dump(self.mape_scaler, os.path.join(path, "mape_scaler.joblib"))
        
        # Save neural network
        torch.save({
            'state_dict': self.nn_model.state_dict(),
            'feature_cols': self.feature_cols,
            'mape_cols': self.mape_cols
        }, os.path.join(path, "mape_predictor.pth"))
        
        # Save forecasting models
        for model_name, model in self.forecasting_models.items():
            model_path = os.path.join(path, f"{model_name}_model.joblib")
            try:
                joblib.dump(model, model_path)
            except Exception as e:
                print(f"Error saving model {model_name}: {str(e)}")
    
    @classmethod
    def load(cls, path: str) -> 'ModelSelector':
        """Load a saved model selector"""
        selector = cls(models_dir=path)
        selector.load_models()
        return selector 