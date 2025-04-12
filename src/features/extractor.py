import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

class TimeSeriesFeatureExtractor:
    def __init__(self):
        # Use minimal feature set for faster computation
        self.feature_params = MinimalFCParameters()
        
    def _detect_frequency(self, timestamps):
        """
        Detect the frequency of the time series from timestamps.
        Returns: 'hourly', 'daily', 'weekly', or 'monthly'
        """
        if isinstance(timestamps[0], str):
            timestamps = pd.to_datetime(timestamps)
        
        # Calculate the most common time delta
        deltas = pd.Series(timestamps).diff().dropna()
        most_common_delta = deltas.mode()[0]
        
        if most_common_delta <= pd.Timedelta(hours=1):
            return 'hourly'
        elif most_common_delta <= pd.Timedelta(days=1):
            return 'daily'
        elif most_common_delta <= pd.Timedelta(days=7):
            return 'weekly'
        else:
            return 'monthly'
    
    def _get_frequency_features(self, timestamps, frequency):
        """Extract frequency-specific features"""
        if isinstance(timestamps[0], str):
            timestamps = pd.to_datetime(timestamps)
            
        features = {}
        
        if frequency == 'hourly':
            features['hour'] = timestamps.hour / 24.0  # Normalize to [0,1]
            features['is_business_hour'] = ((timestamps.hour >= 9) & (timestamps.hour <= 17)).astype(float)
            
        if frequency in ['hourly', 'daily']:
            features['day_of_week'] = timestamps.dayofweek / 6.0  # Normalize to [0,1]
            features['is_weekend'] = (timestamps.dayofweek >= 5).astype(float)
            
        if frequency in ['hourly', 'daily', 'weekly', 'monthly']:
            features['month'] = timestamps.month / 12.0  # Normalize to [0,1]
            features['day_of_month'] = timestamps.day / 31.0  # Normalize to [0,1]
            
        return pd.DataFrame(features)
        
    def extract_features(self, time_series: np.ndarray, timestamps=None) -> pd.DataFrame:
        """
        Extract features from a time series using tsfresh.
        
        Args:
            time_series (np.ndarray): Input time series data
            timestamps (np.ndarray, optional): Timestamp for each point
            
        Returns:
            pd.DataFrame: Extracted features
        """
        # Check for NaN values
        nan_count = pd.isna(time_series).sum()
        if nan_count > 0:
            print(f"Feature extraction: Input contains {nan_count} NaN values out of {len(time_series)} points")
            
        # Step 1: Forward fill, then backward fill (handles internal NaNs)
        clean_ts = pd.Series(time_series).fillna(method='ffill').fillna(method='bfill')
        
        # Step 2: If still have NaNs (e.g., at the beginning/end), use interpolation
        if clean_ts.isna().sum() > 0:
            clean_ts = clean_ts.interpolate(method='linear', limit_direction='both')
        
        # Step 3: Last resort - replace any remaining NaNs with mean or zero
        if clean_ts.isna().sum() > 0:
            mean_val = clean_ts.mean()
            if pd.isna(mean_val):  # If mean is also NaN (all values are NaN)
                mean_val = 0  # Default to zero
            clean_ts = clean_ts.fillna(mean_val)
        
        # Convert to DataFrame format required by tsfresh
        df = pd.DataFrame({
            'id': 0,  # Single time series
            'time': range(len(clean_ts)),
            'value': clean_ts.values
        })
        
        # Extract tsfresh features with error handling
        try:
            features = extract_features(
                df,
                column_id='id',
                column_sort='time',
                column_value='value',
                default_fc_parameters=self.feature_params,
                disable_progressbar=True
            )
            
            # Handle any NaN values in features
            nan_in_features = features.isna().sum().sum()
            if nan_in_features > 0:
                print(f"Feature extraction: Generated features contain {nan_in_features} NaN values")
                
                # Replace NaNs with column means
                column_means = features.mean()
                for col in features.columns:
                    if pd.isna(column_means[col]):
                        column_means[col] = 0  # Use 0 if mean is also NaN
                
                features = features.fillna(column_means)
                
                # Final check - replace any remaining NaNs with 0
                features = features.fillna(0)
                
        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            # Return a DataFrame with zeros for all features
            dummy_ts = np.array([1, 2, 3, 4, 5])
            dummy_features = self.extract_features(dummy_ts)
            features = pd.DataFrame(0, index=[0], columns=dummy_features.columns)
        
        # Add frequency-based features if timestamps are provided
        if timestamps is not None:
            try:
                frequency = self._detect_frequency(timestamps)
                freq_features = self._get_frequency_features(timestamps, frequency)
                
                # Add frequency as one-hot encoding
                freq_one_hot = pd.get_dummies([frequency], prefix='frequency').iloc[0]
                for col in freq_one_hot.index:
                    features[col] = freq_one_hot[col]
                
                # Add mean features for each frequency-specific time period
                for col in freq_features.columns:
                    features[f'mean_by_{col}'] = clean_ts.groupby(freq_features[col]).transform('mean')
                    features[f'std_by_{col}'] = clean_ts.groupby(freq_features[col]).transform('std').fillna(0)
            except Exception as e:
                print(f"Error adding frequency-based features: {str(e)}")
                # Continue without frequency features
        
        # Final verification - make sure no NaNs remain
        if features.isna().sum().sum() > 0:
            features = features.fillna(0)
            
        return features
    
    def get_feature_names(self) -> list:
        """
        Get the names of all features that will be extracted.
        
        Returns:
            list: List of feature names
        """
        # Create a dummy time series with timestamps to get all possible features
        dummy_ts = np.array([1, 2, 3, 4, 5])
        dummy_dates = pd.date_range('2023-01-01', periods=5, freq='D')
        features = self.extract_features(dummy_ts, dummy_dates)
        return list(features.columns) 