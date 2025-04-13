import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_percentage_error

class OfflineTrainer:
    def __init__(self, data_dir, feature_extractor, models):
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        self.models = models

    def load_time_series(self, subdir):
        """Load all time series from a subdirectory"""
        series_list = []
        timestamps_list = []
        path = os.path.join(self.data_dir, subdir)
        if not os.path.exists(path):
            print(f"Warning: Directory {path} does not exist, skipping...")
            return series_list, timestamps_list
            
        for file in os.listdir(path):
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(path, file))
                    if 'point_value' in df.columns and 'point_timestamp' in df.columns:
                        series = df['point_value'].values
                        timestamps = df['point_timestamp'].values
                        # Skip if series is too short
                        if len(series) >= 30:  # Minimum length requirement
                            series_list.append(series)
                            timestamps_list.append(timestamps)
                        else:
                            print(f"Warning: Series in {file} is too short ({len(series)} points), skipping...")
                    else:
                        print(f"Warning: File {file} does not contain required columns, skipping...")
                except Exception as e:
                    print(f"Error loading {file}: {str(e)}")
                    continue
        
        if not series_list:
            print(f"Warning: No valid time series found in {path}")
        else:
            print(f"Loaded {len(series_list)} time series from {path}")
        
        return series_list, timestamps_list

    def prepare_training_data(self):
        """Prepare training data for the neural network"""
        all_features = []
        all_mapes = []
        
        # Process each time frequency
        for subdir in ['daily', 'weekly', 'monthly', 'hourly']:
            print(f"\nProcessing {subdir} data...")
            series_list, timestamps_list = self.load_time_series(subdir)
            
            for ts, timestamps in tqdm(zip(series_list, timestamps_list), desc=f"Extracting features from {subdir}"):
                try:
                    # Extract features with timestamps
                    features = self.feature_extractor.extract_features(ts, timestamps)
                    
                    # Calculate MAPE for each model
                    mapes = []
                    train_size = int(len(ts) * 0.8)
                    train_data = ts[:train_size]
                    test_data = ts[train_size:]
                    
                    for model in self.models:
                        try:
                            model.fit(train_data)
                            predictions = model.predict(len(test_data))
                            mape = self.calculate_mape(test_data, predictions)
                            mapes.append(mape)
                        except Exception as e:
                            print(f"Error with model {model.name}: {str(e)}")
                            mapes.append(float('inf'))
                    
                    # Only add if at least one model worked
                    if not all(np.isinf(mapes)):
                        all_features.append(features.iloc[0].values)
                        all_mapes.append(mapes)
                except Exception as e:
                    print(f"Error processing time series: {str(e)}")
                    continue 

    def calculate_mape(self, actual, predicted):
        """Calculate MAPE with handling for edge cases using sklearn's function"""
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