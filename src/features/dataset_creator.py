import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import mean_absolute_percentage_error

from ..features.extractor import TimeSeriesFeatureExtractor
from ..forecasting.models import ModelFactory

class DatasetCreator:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.feature_extractor = TimeSeriesFeatureExtractor()
        self.models = ModelFactory.get_all_models()
        
    def calculate_mape(self, actual, predicted):
        """Calculate MAPE (Mean Absolute Percentage Error) between actual and predicted values."""
        # Convert inputs to numpy arrays for consistency
        actual = np.asarray(actual)
        predicted = np.asarray(predicted)
        
        # Check for length mismatch
        if len(actual) != len(predicted):
            print(f"Length mismatch in MAPE calculation: actual={len(actual)}, predicted={len(predicted)}")
            return 10.0  # Return 1000% error as a standardized value
        
        # Handle empty arrays
        if len(actual) == 0:
            return 10.0
        
        # Check for NaN or inf values
        if np.any(np.isnan(actual)) or np.any(np.isnan(predicted)) or \
           np.any(np.isinf(actual)) or np.any(np.isinf(predicted)):
            # Try to clean the data
            try:
                # Replace NaN and inf values
                actual = pd.Series(actual).fillna(method='ffill').fillna(method='bfill').fillna(0).values
                predicted = pd.Series(predicted).fillna(method='ffill').fillna(method='bfill').fillna(0).values
            except:
                return 10.0
        
        # Filter out points where actual is zero or close to zero to avoid division by zero
        epsilon = 1e-10
        mask = np.abs(actual) > epsilon
        
        if not np.any(mask):
            return 10.0  # All values are zero or close to zero
        
        actual_filtered = actual[mask]
        predicted_filtered = predicted[mask]
        
        try:
            # Calculate MAPE using sklearn
            mape = mean_absolute_percentage_error(actual_filtered, predicted_filtered)
            
            # Convert to percentage and cap at 1000%
            mape_pct = mape * 100
            result = min(mape_pct, 1000.0)
            
            # Final safety check for NaN
            if np.isnan(result):
                return 10.0
                
            return result
        except Exception as e:
            print(f"Error calculating MAPE: {str(e)}")
            return 10.0  # Standard error value (1000%)
    
    def process_time_series(self, file_path, series_name):
        """Process a single time series file"""
        # Read the data
        df = pd.read_csv(file_path)
        
        # Ensure the file has the correct format with index column
        if 'point_value' not in df.columns:
            print(f"Warning: {series_name} does not contain 'point_value' column, skipping...")
            return None
            
        # Check if the file already has an index column
        if df.columns[0] != 'Unnamed: 0' and df.columns[0] != 'index':
            # Create a new DataFrame with index column
            new_df = pd.DataFrame()
            new_df['index'] = range(len(df))
            new_df['point_timestamp'] = df['point_timestamp'] if 'point_timestamp' in df.columns else None
            new_df['point_value'] = df['point_value']
            
            # Save the reformatted file
            new_df.to_csv(file_path, index=False)
            print(f"Reformatted {series_name} to include index column")
            
            # Use the new DataFrame
            df = new_df
            
        ts = df['point_value'].values
        if len(ts) < 10:  # Minimum length requirement
            print(f"Warning: {series_name} is too short ({len(ts)} points), skipping...")
            return None
            
        try:
            # Extract features
            features = self.feature_extractor.extract_features(ts)
            feature_names = features.columns
            
            # Calculate MAPE for each model
            mapes = {}
            train_size = int(len(ts) * 0.8)
            train_data = ts[:train_size]
            test_data = ts[train_size:]
            
            for model in self.models:
                try:
                    model.fit(train_data)
                    predictions = model.predict(len(test_data))
                    mape = self.calculate_mape(test_data, predictions)
                    mapes[f"MAPE_{model.name}"] = mape
                except Exception as e:
                    print(f"Error with model {model.name} for {series_name}: {str(e)}")
                    mapes[f"MAPE_{model.name}"] = float('inf')
            
            # Create row with series name, features, and MAPE values
            row_data = {'series_name': series_name}
            row_data.update({col: features.iloc[0][col] for col in feature_names})
            row_data.update(mapes)
            
            return row_data
            
        except Exception as e:
            print(f"Error processing {series_name}: {str(e)}")
            return None
    
    def create_dataset(self):
        """Create the dataset from all time series"""
        all_rows = []
        
        # Process each time frequency directory
        for subdir in ['daily', 'weekly', 'monthly', 'hourly']:
            dir_path = os.path.join(self.data_dir, subdir)
            if not os.path.exists(dir_path):
                print(f"Warning: Directory {dir_path} does not exist, skipping...")
                continue
                
            print(f"\nProcessing {subdir} data...")
            for file in tqdm(os.listdir(dir_path)):
                if file.endswith('.csv'):
                    file_path = os.path.join(dir_path, file)
                    series_name = f"{subdir}_{file[:-4]}"  # Remove .csv extension
                    
                    row_data = self.process_time_series(file_path, series_name)
                    if row_data is not None:
                        all_rows.append(row_data)
        
        if not all_rows:
            raise ValueError("No valid time series could be processed")
        
        # Create DataFrame
        df = pd.DataFrame(all_rows)
        
        # Save dataset
        output_path = os.path.join(self.data_dir, 'dataset.csv')
        df.to_csv(output_path, index=False)
        print(f"\nDataset saved to {output_path}")
        print(f"Total series processed: {len(df)}")
        print("\nFeature columns:", ', '.join(df.columns[1:-len(self.models)]))  # Exclude series_name and MAPE columns
        print("MAPE columns:", ', '.join(df.columns[-len(self.models):]))
        
        return df

def main():
    creator = DatasetCreator()
    print("Creating dataset...")
    df = creator.create_dataset()
    print("\nDataset creation complete!") 