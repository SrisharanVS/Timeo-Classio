import os
import sys
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import mean_absolute_percentage_error

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.classifier.model_selector import ModelSelector

def evaluate_model(actual, predicted):
    """Evaluate model performance"""
    try:
        # Filter out NaN values
        mask = ~np.isnan(actual) & ~np.isnan(predicted)
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            print("Warning: No valid data points for evaluation after filtering NaNs")
            return None
            
        # Calculate MAPE
        mape = mean_absolute_percentage_error(actual_clean, predicted_clean) * 100
        print(f"\nModel Evaluation:")
        print(f"MAPE: {mape:.2f}%")
        return mape
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Predict using the best model for a time series')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV file containing time series data')
    parser.add_argument('--models-dir', type=str, default='models', help='Directory containing trained models')
    parser.add_argument('--test-size', type=float, default=0.2, help='Fraction of data to use as test set')
    parser.add_argument('--output', type=str, help='Output file for predictions')
    parser.add_argument('--model', type=str, help='Force using a specific model instead of selecting best')
    args = parser.parse_args()

    # Load time series data
    try:
        # Read CSV file with pandas
        df = pd.read_csv(args.data)
        # Extract the point_value column and convert to numpy array
        data = df['point_value'].to_numpy()
        
        # Check for timestamps
        has_timestamps = 'point_timestamp' in df.columns
        timestamps = df['point_timestamp'].values if has_timestamps else None
        
        print(f"Loaded time series with {len(data)} points")
    except Exception as e:
        print(f"Error: Could not load data file: {str(e)}")
        print("Please ensure the CSV file has a 'point_value' column.")
        return

    # Initialize model selector
    try:
        selector = ModelSelector(models_dir=args.models_dir)
    except Exception as e:
        print(f"Error initializing model selector: {str(e)}")
        return

    # Split data into train and test sets
    train_size = int(len(data) * (1 - args.test_size))
    train_data = data[:train_size]
    test_data = data[train_size:]
    train_timestamps = timestamps[:train_size] if timestamps is not None else None
    test_timestamps = timestamps[train_size:] if timestamps is not None else None
    
    print(f"\nSplit data into:")
    print(f"Training set: {len(train_data)} points")
    print(f"Test set: {len(test_data)} points")
    
    try:
        if args.model:
            # Use specified model
            best_model_name = args.model
            model = selector.get_model(best_model_name)
            predicted_mape = None
            print(f"\nUsing specified model: {best_model_name}")
        else:
            # Select best model using only training data
            best_model_name, predicted_mape = selector.select_model(train_data)
            model = selector.get_model(best_model_name)
            print(f"\nModel Selection Results:")
            print(f"Best model: {best_model_name}")
            if predicted_mape is not None:
                print(f"Predicted MAPE: {predicted_mape:.4f}")

        if model is None:
            print(f"Error: Could not load model {best_model_name}")
            return

        # Train on training data and predict test data
        model.fit(train_data)
        test_predictions = model.predict(len(test_data))
        
        # Calculate MAPE on test data
        test_mape = evaluate_model(test_data, test_predictions)
        
        if predicted_mape is not None and test_mape is not None:
            print(f"MAPE prediction error: {abs(predicted_mape - test_mape):.2f}%")
        
        # Print actual vs predicted values for test set
        print("\nTest Set Predictions:")
        print("Timestamp".ljust(25) if has_timestamps else "Index".ljust(10), 
              "Actual".rjust(15), 
              "Predicted".rjust(15), 
              "Error %".rjust(10))
        print("-" * (65 if has_timestamps else 50))
        
        for i in range(len(test_data)):
            actual = test_data[i]
            pred = test_predictions[i]
            
            # Handle NaN values in error calculation
            if np.isnan(actual) or np.isnan(pred):
                error = float('nan')
            else:
                error = abs(actual - pred) / (abs(actual) if abs(actual) > 1e-10 else 1) * 100
            
            if has_timestamps:
                print(f"{test_timestamps[i]:<25} {actual:>15.2f} {pred:>15.2f} {error:>10.2f}")
            else:
                print(f"{(train_size + i):<10d} {actual:>15.2f} {pred:>15.2f} {error:>10.2f}")
        
        # Save predictions if output file is specified
        if args.output:
            output_df = pd.DataFrame({
                'timestamp': test_timestamps if has_timestamps else range(train_size, len(data)),
                'actual': test_data,
                'predicted': test_predictions,
                'error_percent': [
                    float('nan') if np.isnan(actual) or np.isnan(pred) else
                    abs(actual - pred) / (abs(actual) if abs(actual) > 1e-10 else 1) * 100
                    for actual, pred in zip(test_data, test_predictions)
                ]
            })
            output_df.to_csv(args.output, index=False)
            print(f"\nSaved predictions to {args.output}")
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == '__main__':
    main() 