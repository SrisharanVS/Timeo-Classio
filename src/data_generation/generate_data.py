import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

def generate_daily_data(n_series=10, days=365*2):
    """Generate daily time series with different patterns."""
    series_list = []
    
    for i in range(n_series):
        # Create time index
        end_date = datetime.now().date()
        dates = [(end_date - timedelta(days=x)) for x in range(days)]
        dates.reverse()
        
        # Base trend
        trend = np.linspace(50, 100, days) * (1 + np.random.uniform(-0.2, 0.2))
        
        # Add yearly seasonality
        yearly = 20 * np.sin(2 * np.pi * np.arange(days) / 365)
        
        # Add weekly seasonality
        weekly = 10 * np.sin(2 * np.pi * np.arange(days) / 7)
        
        # Add noise
        noise = np.random.normal(0, 5, days)
        
        # Combine components
        values = trend + yearly + weekly + noise
        
        # Ensure all values are positive
        values = np.maximum(values, 1)
        
        # Create DataFrame
        df = pd.DataFrame({
            'point_timestamp': dates,
            'point_value': values
        })
        
        # Save to CSV
        os.makedirs('data/daily', exist_ok=True)
        filename = f'data/daily/series_{i+1}.csv'
        df.to_csv(filename, index=False)
        series_list.append(filename)
    
    return series_list

def generate_weekly_data(n_series=10, weeks=52*2):
    """Generate weekly time series with different patterns."""
    series_list = []
    
    for i in range(n_series):
        # Create time index
        end_date = datetime.now().date()
        dates = [(end_date - timedelta(weeks=x)) for x in range(weeks)]
        dates.reverse()
        
        # Base trend
        trend = np.linspace(100, 200, weeks) * (1 + np.random.uniform(-0.3, 0.3))
        
        # Add yearly seasonality
        yearly = 30 * np.sin(2 * np.pi * np.arange(weeks) / 52)
        
        # Add noise
        noise = np.random.normal(0, 10, weeks)
        
        # Combine components
        values = trend + yearly + noise
        
        # Ensure all values are positive
        values = np.maximum(values, 1)
        
        # Create DataFrame
        df = pd.DataFrame({
            'point_timestamp': dates,
            'point_value': values
        })
        
        # Save to CSV
        os.makedirs('data/weekly', exist_ok=True)
        filename = f'data/weekly/series_{i+1}.csv'
        df.to_csv(filename, index=False)
        series_list.append(filename)
    
    return series_list

def generate_monthly_data(n_series=10, months=24):
    """Generate monthly time series with different patterns."""
    series_list = []
    
    for i in range(n_series):
        # Create time index
        end_date = datetime.now().date().replace(day=1)
        dates = [(end_date - pd.DateOffset(months=x)) for x in range(months)]
        dates.reverse()
        
        # Base trend
        trend = np.linspace(500, 1000, months) * (1 + np.random.uniform(-0.4, 0.4))
        
        # Add yearly seasonality
        yearly = 100 * np.sin(2 * np.pi * np.arange(months) / 12)
        
        # Add noise
        noise = np.random.normal(0, 20, months)
        
        # Combine components
        values = trend + yearly + noise
        
        # Ensure all values are positive
        values = np.maximum(values, 1)
        
        # Create DataFrame
        df = pd.DataFrame({
            'point_timestamp': dates,
            'point_value': values
        })
        
        # Save to CSV
        os.makedirs('data/monthly', exist_ok=True)
        filename = f'data/monthly/series_{i+1}.csv'
        df.to_csv(filename, index=False)
        series_list.append(filename)
    
    return series_list

def generate_hourly_data(n_series=10, hours=24*30):
    """Generate hourly time series with different patterns."""
    series_list = []
    
    for i in range(n_series):
        # Create time index
        end_date = datetime.now().replace(minute=0, second=0, microsecond=0)
        dates = [(end_date - timedelta(hours=x)) for x in range(hours)]
        dates.reverse()
        
        # Base trend
        trend = np.linspace(20, 40, hours) * (1 + np.random.uniform(-0.1, 0.1))
        
        # Add daily seasonality
        daily = 10 * np.sin(2 * np.pi * np.arange(hours) / 24)
        
        # Add weekly seasonality
        weekly = 5 * np.sin(2 * np.pi * np.arange(hours) / (24 * 7))
        
        # Add noise
        noise = np.random.normal(0, 2, hours)
        
        # Combine components
        values = trend + daily + weekly + noise
        
        # Ensure all values are positive
        values = np.maximum(values, 1)
        
        # Create DataFrame
        df = pd.DataFrame({
            'point_timestamp': dates,
            'point_value': values
        })
        
        # Save to CSV
        os.makedirs('data/hourly', exist_ok=True)
        filename = f'data/hourly/series_{i+1}.csv'
        df.to_csv(filename, index=False)
        series_list.append(filename)
    
    return series_list

if __name__ == "__main__":
    print("Generating synthetic time series data...")
    
    # Generate data for each frequency
    daily_files = generate_daily_data(n_series=10)
    print(f"Generated {len(daily_files)} daily time series")
    
    weekly_files = generate_weekly_data(n_series=10)
    print(f"Generated {len(weekly_files)} weekly time series")
    
    monthly_files = generate_monthly_data(n_series=10)
    print(f"Generated {len(monthly_files)} monthly time series")
    
    hourly_files = generate_hourly_data(n_series=10)
    print(f"Generated {len(hourly_files)} hourly time series")
    
    print("\nData generation complete. Files are saved in data/{daily,weekly,monthly,hourly} directories.") 