import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error
from prophet import Prophet
import os

class TimeSeriesModel:
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.mape = None
        
    def fit(self, data: np.ndarray):
        """Fit the model to the data"""
        raise NotImplementedError
        
    def predict(self, steps: int) -> np.ndarray:
        """Generate predictions for the specified number of steps"""
        raise NotImplementedError
        
    def calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate MAPE for the predictions using sklearn's function with edge case handling"""
        # Convert inputs to numpy arrays
        actual = np.asarray(actual)
        predicted = np.asarray(predicted)
        
        # Check for length mismatch
        if len(actual) != len(predicted):
            print(f"Length mismatch in MAPE calculation: actual={len(actual)}, predicted={len(predicted)}")
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
            result = min(mape, 1000.0)  # Cap at 1000%
            
            # Final safety check for NaN
            if np.isnan(result):
                return float('inf')
                
            self.mape = result
            return result
        except Exception as e:
            print(f"Error calculating MAPE: {str(e)}")
            return float('inf')

    def __getstate__(self):
        """Return state values to be pickled."""
        return self.__dict__
        
    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.__dict__ = state

class ARIMAModel(TimeSeriesModel):
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """
        Initialize ARIMA model.
        
        Args:
            order: The (p,d,q) order of the model for the number of AR parameters,
                  differences, and MA parameters.
            seasonal_order: The (P,D,Q,s) order of the seasonal component of the model
                          for the AR parameters, differences, MA parameters, and periodicity.
        """
        super().__init__("ARIMA")
        self.order = order
        self.seasonal_order = seasonal_order
        self.data = None
        self.seasonal = False
        self.seasonality_period = None
        self.fitted = False
        
    def _detect_seasonality(self, data):
        """Detect if the series has seasonality and determine the period"""
        if len(data) < 20:
            return False, None
            
        # Use autocorrelation to detect seasonality
        from statsmodels.tsa.stattools import acf
        
        # Calculate autocorrelation for different lags
        acf_values = acf(data, nlags=min(len(data) // 2, 50))
        
        # Look for peaks in the ACF
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(acf_values, height=0.3)  # Find peaks with height > 0.3
        
        if len(peaks) > 1:
            # Get the first major peak after lag 1
            for peak in peaks:
                if peak > 1:  # Skip the first peak at lag 0
                    self.seasonality_period = peak
                    return True, peak
        
        return False, None
    
    def _determine_best_orders(self, data):
        """Use AIC to determine the best ARIMA orders"""
        import itertools
        best_aic = float('inf')
        best_order = (1, 1, 1)
        best_seasonal_order = (0, 0, 0, 0)
        
        # Limit the search space for performance
        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        
        if self.seasonal and self.seasonality_period:
            P = D = Q = range(0, 2)
            seasonal_pdq = [(x[0], x[1], x[2], self.seasonality_period) for x in list(itertools.product(P, D, Q))]
        else:
            seasonal_pdq = [(0, 0, 0, 0)]
        
        # Try a limited set of models for performance
        from statsmodels.tsa.arima.model import ARIMA
        import warnings
        warnings.filterwarnings('ignore')
        
        # Use only a subset of the data for model selection
        test_data = data[-min(len(data), 100):]
        
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    if self.seasonal:
                        # Use SARIMAX for seasonal models
                        from statsmodels.tsa.statespace.sarimax import SARIMAX
                        model = SARIMAX(test_data, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
                    else:
                        model = ARIMA(test_data, order=param)
                    
                    model_fit = model.fit(disp=0)
                    aic = model_fit.aic
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_order = param
                        best_seasonal_order = param_seasonal
                except:
                    continue
        
        return best_order, best_seasonal_order
        
    def fit(self, data: np.ndarray):
        self.data = data
        
        # Detect seasonality
        self.seasonal, self.seasonality_period = self._detect_seasonality(data)
        
        # Determine best model orders
        if self.order is None:
            self.order, self.seasonal_order = self._determine_best_orders(data)
        
        # Fit the model
        if self.seasonal and self.seasonality_period:
            # Use SARIMAX for seasonal models
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            self.model = SARIMAX(
                data, 
                order=self.order, 
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            try:
                self.model = self.model.fit(disp=0)
            except TypeError:
                # Fall back if disp is not supported
                self.model = self.model.fit()
        else:
            # Use regular ARIMA for non-seasonal models
            from statsmodels.tsa.arima.model import ARIMA
            self.model = ARIMA(data, order=self.order)
            try:
                self.model = self.model.fit(disp=0)
            except TypeError:
                # Fall back if disp is not supported
                self.model = self.model.fit()
        
        self.fitted = True
        
    def predict(self, steps: int) -> np.ndarray:
        return self.model.forecast(steps=steps)

    def __getstate__(self):
        """Return state values to be pickled."""
        state = self.__dict__.copy()
        # Ensure seasonal_order is included in the state
        if 'seasonal_order' not in state:
            state['seasonal_order'] = (1, 1, 1, 12)  # Default value
        return state
        
    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.__dict__ = state
        # Ensure seasonal_order is available
        if not hasattr(self, 'seasonal_order'):
            self.seasonal_order = (1, 1, 1, 12)  # Default value

class ExponentialSmoothingModel(TimeSeriesModel):
    def __init__(self):
        super().__init__("ExponentialSmoothing")
        self.seasonal_periods = None
        
    def _detect_seasonality(self, data):
        """Detect the seasonal period in the data"""
        if len(data) < 20:
            return 1  # Default to no seasonality for short series
            
        # Try common seasonality periods
        if len(data) > 365:
            return 365  # Daily data with annual seasonality
        elif len(data) > 52:
            return 52   # Weekly data with annual seasonality
        elif len(data) > 30:
            return 30   # Daily data with monthly seasonality
        elif len(data) > 24:
            return 24   # Hourly data with daily seasonality
        elif len(data) > 12:
            return 12   # Monthly data with annual seasonality
        elif len(data) > 7:
            return 7    # Daily data with weekly seasonality
        else:
            return 1    # No detected seasonality
        
    def fit(self, data: np.ndarray):
        # Check if there are any NaN or inf values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            # Clean the data
            clean_data = np.array(data, dtype=float)
            clean_data = pd.Series(clean_data).fillna(method='ffill').fillna(method='bfill').values
        else:
            clean_data = data
            
        # Detect seasonality
        self.seasonal_periods = self._detect_seasonality(clean_data)
        
        # Only use seasonality if period > 1 and we have enough data
        use_seasonal = self.seasonal_periods > 1 and len(clean_data) >= 2 * self.seasonal_periods
        
        try:
            if use_seasonal:
                # Try seasonal model first
                try:
                    self.model = ExponentialSmoothing(
                        clean_data,
                        seasonal_periods=self.seasonal_periods,
                        trend='add',
                        seasonal='add',
                        damped_trend=True
                    )
                    self.model = self.model.fit(optimized=True)
                    return
                except Exception as e:
                    print(f"Warning: Seasonal model failed, falling back to non-seasonal: {str(e)}")
                    # If seasonal fails, fall back to non-seasonal
                    
            # Non-seasonal model
            self.model = ExponentialSmoothing(
                clean_data,
                trend='add',
                seasonal=None,
                damped_trend=True
            )
            self.model = self.model.fit(optimized=True)
        except Exception as e:
            # Fall back to even simpler model if all else fails
            print(f"Warning: Exponential smoothing failed, using simple exponential smoothing: {str(e)}")
            from statsmodels.tsa.holtwinters import SimpleExpSmoothing
            self.model = SimpleExpSmoothing(clean_data).fit()
        
    def predict(self, steps: int) -> np.ndarray:
        return self.model.forecast(steps)

class SimpleMovingAverageModel(TimeSeriesModel):
    def __init__(self, min_window: int = 3, max_window: int = 10):
        super().__init__("SimpleMovingAverage")
        self.min_window = min_window
        self.max_window = max_window
        self.window_size = None
        self.data = None
        self.trend = 0
        
    def _optimize_window_size(self, data: np.ndarray) -> int:
        """Find optimal window size using cross-validation on recent data."""
        # Ensure window sizes are valid
        self.min_window = max(3, self.min_window)
        self.max_window = min(len(data) // 2, self.max_window)
        self.max_window = max(self.min_window, self.max_window)
        
        # If data is too short, use minimum window
        if len(data) < 2 * self.min_window:
            return self.min_window
        
        best_window = self.min_window
        best_error = float('inf')
        
        # Use the last 30% of data for validation
        train_size = int(len(data) * 0.7)
        train_size = max(self.max_window + 1, train_size)  # Ensure we have enough training data
        train_data = data[:train_size]
        val_data = data[train_size:]
        
        # If validation set is too small, use minimum window
        if len(val_data) < 3:
            return self.min_window
        
        for window in range(self.min_window, min(self.max_window + 1, len(train_data))):
            predictions = []
            for i in range(len(val_data)):
                # Get the window of history before this point
                hist_idx_start = min(train_size - window + i, len(data) - window)
                hist_idx_end = min(train_size + i, len(data))
                hist = data[hist_idx_start:hist_idx_end]
                
                if len(hist) == 0:  # Skip if no history
                    continue
                
                pred = np.mean(hist)
                predictions.append(pred)
            
            # Skip if no predictions
            if not predictions:
                continue
                
            # Truncate validation data if needed
            val_subset = val_data[:len(predictions)]
            
            # Calculate error - handle zeros by adding a small constant
            try:
                errors = np.abs((val_subset - predictions) / (val_subset + 1e-8))
                error = np.mean(errors)
            except:
                # Fallback error calculation
                error = np.mean(np.abs(val_subset - predictions))
            
            if error < best_error:
                best_error = error
                best_window = window
        
        return best_window
    
    def _estimate_trend(self, data: np.ndarray) -> float:
        """Estimate the trend using linear regression."""
        if len(data) < 2:
            return 0
            
        x = np.arange(len(data))
        # Add small constant to prevent numerical issues
        y = data + 1e-8
        
        try:
            coeffs = np.polyfit(x, y, 1)
            return coeffs[0]  # Return the slope
        except:
            # Fallback if polyfit fails
            if len(data) >= 2:
                # Simple slope calculation
                return (data[-1] - data[0]) / (len(data) - 1)
            return 0
        
    def fit(self, data: np.ndarray):
        # Handle NaN or Inf values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            clean_data = pd.Series(data).fillna(method='ffill').fillna(method='bfill').values
        else:
            clean_data = data
            
        self.data = clean_data
        
        # Optimize window size
        try:
            self.window_size = self._optimize_window_size(clean_data)
        except Exception as e:
            print(f"Warning: Failed to optimize window size: {str(e)}")
            self.window_size = min(10, max(3, len(clean_data) // 5))
        
        # Estimate trend
        try:
            self.trend = self._estimate_trend(clean_data[-self.window_size:])
        except Exception as e:
            print(f"Warning: Failed to estimate trend: {str(e)}")
            self.trend = 0
        
    def predict(self, steps: int) -> np.ndarray:
        if self.data is None:
            raise ValueError("Model must be fit before predicting")
        
        if len(self.data) == 0:
            return np.zeros(steps)
            
        # Get last values for prediction
        window_size = min(self.window_size, len(self.data))
        last_values = self.data[-window_size:].copy()
        
        # Check for NaN values in last_values and clean them
        if np.any(np.isnan(last_values)):
            print(f"SimpleMovingAverage: Last values contain {np.sum(np.isnan(last_values))} NaN values")
            
            # Convert to pandas Series for easier handling
            last_values_series = pd.Series(last_values)
            # Forward fill then backward fill
            last_values_series = last_values_series.fillna(method='ffill').fillna(method='bfill')
            
            # If still have NaNs, use zeros
            if last_values_series.isna().sum() > 0:
                last_values_series = last_values_series.fillna(0)
                
            last_values = last_values_series.values
            
        try:
            predictions = []
            for _ in range(steps):
                # Calculate base prediction using moving average
                base_pred = np.nanmean(last_values)  # Use nanmean to handle any remaining NaNs
                
                # If base_pred is NaN, use the last valid prediction or 0
                if np.isnan(base_pred):
                    base_pred = predictions[-1] if predictions else 0
                
                # Add trend component
                trend_adjustment = self.trend
                next_value = base_pred + trend_adjustment
                
                predictions.append(next_value)
                
                # Update last values for next prediction
                if len(last_values) > 1:
                    last_values = np.roll(last_values, -1)
                    last_values[-1] = next_value
                else:
                    last_values[0] = next_value
                
            # Final check for NaNs in predictions
            if np.any(np.isnan(predictions)):
                print(f"SimpleMovingAverage: Predictions contain {np.sum(np.isnan(predictions))} NaN values")
                # Fall back to a constant prediction using the mean of the original data
                data_mean = np.nanmean(self.data)
                if np.isnan(data_mean):
                    data_mean = 0
                predictions = np.full(steps, data_mean)
                
            return np.array(predictions)
            
        except Exception as e:
            print(f"SimpleMovingAverage prediction error: {str(e)}")
            # Return a constant prediction as fallback
            fallback_value = np.nanmean(self.data) if len(self.data) > 0 else 0
            if np.isnan(fallback_value):
                fallback_value = 0
            return np.full(steps, fallback_value)

class ProphetModel(TimeSeriesModel):
    def __init__(self):
        super().__init__("Prophet")
        self.model = None
        self.last_date = None
        self.freq = None
        
    def _detect_frequency(self, dates=None):
        """Detect the frequency of the time series data"""
        if dates is None or len(dates) < 2:
            return 'D'  # Default to daily
            
        # Convert to pandas datetime if not already
        dates = pd.to_datetime(dates)
        
        # Calculate the most common time difference
        diff = pd.Series(dates).diff().value_counts().index[0]
        
        # Convert timedelta to string frequency
        if diff.days == 0:
            if diff.seconds == 3600:  # 1 hour
                return 'H'
            elif diff.seconds == 1800:  # 30 minutes
                return '30min'
            elif diff.seconds == 900:  # 15 minutes
                return '15min'
            else:
                return 'D'  # Default to daily for sub-daily data
        elif diff.days == 1:
            return 'D'
        elif diff.days == 7:
            return 'W'
        elif 28 <= diff.days <= 31:
            return 'M'
        else:
            return 'D'  # Default to daily
        
    def fit(self, data: np.ndarray, dates=None):
        """
        Fit Prophet model to the data.
        
        Args:
            data: Array of time series values
            dates: Optional array of dates. If not provided, will create synthetic dates.
        """
        # Create DataFrame for Prophet
        if dates is not None:
            # Convert to pandas datetime and ensure consistent format
            dates = pd.to_datetime(dates)
            # Round to nearest hour if seconds are present
            dates = dates.round('H')
            self.freq = self._detect_frequency(dates)
            self.last_date = dates[-1]
        else:
            # Create synthetic dates if not provided
            self.freq = 'D'
            self.last_date = pd.Timestamp.now()
            dates = pd.date_range(end=self.last_date, periods=len(data), freq=self.freq)
            
        # Create DataFrame and handle missing values
        df = pd.DataFrame({
            'ds': dates,
            'y': data
        })
        
        # Remove rows with NaN values
        df = df.dropna()
        
        if len(df) < 2:
            raise ValueError("Not enough valid data points after removing NaN values")
            
        try:
            # Initialize and fit Prophet model with robust settings
            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True if self.freq == 'H' else False,
                seasonality_mode='multiplicative',
                interval_width=0.95,
                uncertainty_samples=1000
            )
            
            with suppress_stdout_stderr():
                self.model.fit(df)
                
        except Exception as e:
            print(f"Error fitting Prophet model: {str(e)}")
            # Fall back to simpler model settings
            try:
                self.model = Prophet(
                    yearly_seasonality=False,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode='additive',
                    interval_width=0.95,
                    uncertainty_samples=100
                )
                with suppress_stdout_stderr():
                    self.model.fit(df)
            except Exception as e:
                print(f"Error fitting simplified Prophet model: {str(e)}")
                raise
        
    def predict(self, steps: int) -> np.ndarray:
        """Generate predictions for the specified number of steps"""
        if self.model is None:
            raise ValueError("Model must be fit before predicting")
            
        # Create future dates DataFrame
        future_dates = pd.date_range(
            start=self.last_date,
            periods=steps + 1,  # +1 because we'll exclude the start date
            freq=self.freq
        )[1:]  # Exclude the start date
        
        future = pd.DataFrame({'ds': future_dates})
        
        try:
            # Make predictions
            with suppress_stdout_stderr():
                forecast = self.model.predict(future)
            return forecast['yhat'].values
        except Exception as e:
            print(f"Error making Prophet predictions: {str(e)}")
            # Return last value repeated
            return np.full(steps, self.data[-1] if hasattr(self, 'data') else 0)

class suppress_stdout_stderr:
    """Context manager to suppress stdout and stderr"""
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

class ModelFactory:
    @staticmethod
    def get_all_models():
        """Get a list of all available forecasting models"""
        return [
            ARIMAModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)),
            ExponentialSmoothingModel(),
            SimpleMovingAverageModel(),
            ProphetModel()
        ]
        
    @staticmethod
    def get_model(name: str):
        """Get a specific model by name"""
        name = name.lower()
        if name == 'arima':
            return ARIMAModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        elif name == 'exponentialsmoothing':
            return ExponentialSmoothingModel()
        elif name == 'simplemovingaverage':
            return SimpleMovingAverageModel()
        elif name == 'prophet':
            return ProphetModel()
        else:
            raise ValueError(f"Unknown model: {name}") 