# Time Series Forecasting System

## DEMO

![image](https://github.com/user-attachments/assets/677e8245-fd38-4a80-afaf-962fea84c078)

![image](https://github.com/user-attachments/assets/7e91df0c-d0a1-47c5-b693-5896539e7ec3)




## How the Classifier Was Trained: My Perspective

### The Challenge

The core challenge was to automatically select the best forecasting model for any given time series. The traditional approach of trying each model and comparing results would be computationally expensive and impractical for real-time applications. I needed a more efficient solution that could predict which model would perform best without actually running all models.

### Feature Engineering Approach

I implemented a comprehensive feature extraction system that captures the essential characteristics of time series data:

1. **Statistical Features**: I used the tsfresh library with MinimalFCParameters to extract statistical features efficiently. This approach balances feature richness with computational performance.
   
2. **Temporal Features**: I implemented frequency-specific feature extraction that adapts to the data granularity (hourly, daily, weekly, monthly). This includes:
   - Hourly data: hour of day, business hours indicator
   - Daily data: day of week, weekend indicator
   - All frequencies: month, day of month

3. **Data Quality Handling**: I added robust preprocessing to handle missing values through:
   - Forward fill and backward fill for internal NaNs
   - Linear interpolation for edge cases
   - Mean imputation as a last resort

### Neural Network Architecture

For the neural network architecture, I implemented the MAPEPredictor with these specific design choices:

1. **Batch Normalization**: I added batch normalization after each layer to improve training stability and convergence.

2. **Dropout**: I implemented dropout (0.2) to prevent overfitting, which is crucial when working with limited training data.

3. **Layer Structure**: I used two fully connected layers with 64 neurons each, followed by an output layer with 4 neurons (one for each forecasting model).

### Training Process Implementation

The actual training process follows these specific steps:

1. **Data Preparation**: 
   - Split data into training (80%) and validation (20%) sets
   - Scale features and MAPE values using StandardScaler
   - Convert to PyTorch tensors for efficient processing

2. **Training Configuration**:
   - Adam optimizer with learning rate of 0.001
   - MSE loss function
   - Batch size of 32
   - Early stopping with patience of 10 epochs

3. **Training Loop**:
   - Train for up to 100 epochs
   - Monitor validation loss for early stopping
   - Save the best model state based on validation performance

### Model Selection Process

The model selection process works as follows:

1. Extract features from the input time series using TimeSeriesFeatureExtractor
2. Scale features using the saved feature_scaler
3. Predict MAPE for each model using the trained neural network
4. Select the model with the lowest predicted MAPE

### Performance Optimization

To ensure efficient operation, I implemented several optimizations:

1. **Feature Extraction Efficiency**: Using tsfresh's MinimalFCParameters for faster computation
2. **Memory Management**: Efficient data loading and processing to handle large datasets
3. **Early Stopping**: Preventing overfitting while minimizing training time

### Validation Strategy

To ensure the classifier's reliability, I implemented:

1. **Train-Validation Split**: 80% training, 20% validation
2. **Early Stopping**: Prevents overfitting by monitoring validation loss
3. **Best Model Selection**: Saves the model state with the lowest validation loss

This implementation approach balances model performance with computational efficiency, making it suitable for real-world time series forecasting applications. 




## Project Overview

This project is a comprehensive time series forecasting solution that:

1. Automatically selects the best forecasting model for your time series data
2. Supports multiple forecasting models (ARIMA, Exponential Smoothing, Simple Moving Average, Prophet)
3. Uses machine learning to predict model performance
4. Provides both REST API and web interface for easy interaction
5. Visualizes forecasts using Plotly

### Key Features

- **Automatic Model Selection**: Uses a neural network to predict the best forecasting model for your data
- **Multiple Forecasting Models**: Supports ARIMA, Exponential Smoothing, Simple Moving Average, and Prophet
- **Interactive Web Interface**: Upload data and visualize forecasts with Plotly
- **REST API**: Programmatic access to forecasting capabilities
- **Real-time Performance Metrics**: MAPE (Mean Absolute Percentage Error) calculation and visualization

## Technical Architecture

The project is structured into several key components:

1. **Forecasting Models** (`src/forecasting/`):
   - ARIMA Model
   - Exponential Smoothing Model
   - Simple Moving Average Model
   - Prophet Model

2. **Model Selection** (`src/classifier/`):
   - Neural network-based model selector
   - Feature extraction and preprocessing
   - Model performance prediction

3. **Web Interface** (`src/web/`):
   - FastAPI backend
   - Interactive Plotly visualizations
   - File upload and data processing

4. **Feature Engineering** (`src/features/`):
   - Time series feature extraction
   - Dataset creation and preprocessing

## Setup and Installation

### Prerequisites

- Python 3.10 (using pyenv)
- pip (Python package manager)

### Environment Setup

1. Install Python 3.10 using pyenv:
   ```bash
   pyenv install 3.10
   pyenv local 3.10
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Generation and Model Training

1. Generate the dataset for neural network training:
   ```bash
   python -c "from src.features.dataset_creator import main; main()"
   ```

2. Train the model selector:
   ```bash
   python src/classifier/train_model_selector.py data/dataset.csv --models-dir models
   ```

## API Documentation

### REST API Endpoints

1. **POST /predict**
   - Upload a CSV file with time series data
   - Returns the best model and forecast
   - Required columns: `point_value` (optional: `point_timestamp`)

2. **GET /**
   - Web interface for interactive usage
   - Upload data and visualize forecasts

### Web Interface

The web interface provides:
- File upload for time series data
- Interactive Plotly visualizations
- Real-time model selection
- Performance metrics display

## Usage Examples

### Using the REST API

```python
import requests
import pandas as pd

# Prepare your data
data = pd.DataFrame({
    'point_value': [1, 2, 3, 4, 5]
})

# Save to CSV
data.to_csv('data.csv', index=False)

# Make prediction request
with open('data.csv', 'rb') as f:
    response = requests.post('http://localhost:8000/predict', files={'file': f})
    
print(response.json())
```

### Using the Web Interface

1. Start the web server:
   ```bash
   uvicorn src.web.app:app --reload
   ```

2. Open your browser and navigate to `http://localhost:8000`
3. Upload your time series data
4. View the forecast and model selection results

## Technical Details

### Model Selection Process

1. **Feature Extraction**:
   - Statistical features (mean, variance, etc.)
   - Time series characteristics (trend, seasonality)
   - Data quality metrics

2. **Neural Network Architecture**:
   - Input layer: Time series features
   - Hidden layers: Multiple fully connected layers
   - Output layer: MAPE predictions for each model

3. **Model Training**:
   - Dataset split: 80% training, 20% validation
   - Early stopping to prevent overfitting
   - Batch training with Adam optimizer

### Forecasting Models

1. **ARIMA**:
   - Automatic order selection
   - Seasonal decomposition
   - Robust error handling

2. **Exponential Smoothing**:
   - Multiple seasonality support
   - Automatic parameter optimization
   - Trend and seasonality detection

3. **Simple Moving Average**:
   - Adaptive window size
   - Trend estimation
   - Robust to outliers

4. **Prophet**:
   - Automatic seasonality detection
   - Holiday effects
   - Robust to missing data

## Classifier Training Process

### Feature Engineering for Classification

The classifier uses a comprehensive set of features extracted from time series data:

1. **Statistical Features** (via tsfresh):
   - Basic statistics: mean, variance, skewness, kurtosis
   - Distribution characteristics: quantiles, percentiles
   - Trend indicators: linear trend, exponential trend
   - Seasonality measures: autocorrelation, partial autocorrelation
   - Stationarity tests: Augmented Dickey-Fuller test
   - Entropy measures: sample entropy, approximate entropy

2. **Temporal Features**:
   - Frequency-specific features based on data granularity:
     - Hourly: hour of day, business hours indicator
     - Daily: day of week, weekend indicator
     - Weekly: week of month
     - Monthly: month, season
   - Time-based patterns: day of month, quarter, year
   - Holiday and special day indicators

3. **Data Quality Metrics**:
   - Missing value ratio
   - Outlier proportion
   - Data completeness
   - Sampling consistency

### Neural Network Architecture

The classifier implements a deep neural network (`MAPEPredictor`) with the following architecture:

1. **Input Layer**:
   - Batch normalization for feature standardization
   - Input size: Number of extracted features

2. **Hidden Layers**:
   - Two fully connected layers with 64 neurons each
   - Batch normalization after each layer
   - ReLU activation function
   - Dropout (0.2) for regularization

3. **Output Layer**:
   - 4 neurons (one for each forecasting model)
   - Predicts MAPE values for each model

### Training Process

1. **Data Preparation**:
   ```python
   # Split data into training and validation sets
   X_train, X_val, y_train, y_val = train_test_split(
       features, targets, test_size=0.2, random_state=42
   )
   
   # Scale features and targets
   feature_scaler = StandardScaler()
   mape_scaler = StandardScaler()
   
   X_train_scaled = feature_scaler.fit_transform(X_train)
   X_val_scaled = feature_scaler.transform(X_val)
   ```

2. **Model Training**:
   ```python
   # Initialize model
   model = MAPEPredictor(input_size=len(feature_cols))
   
   # Training configuration
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   criterion = nn.MSELoss()
   
   # Early stopping
   early_stopping = EarlyStopping(patience=10)
   
   # Training loop
   for epoch in range(epochs):
       model.train()
       for batch in train_loader:
           features, targets = batch
           optimizer.zero_grad()
           predictions = model(features)
           loss = criterion(predictions, targets)
           loss.backward()
           optimizer.step()
   ```

3. **Model Selection Process**:
   - For each time series:
     1. Extract features using `TimeSeriesFeatureExtractor`
     2. Scale features using saved `feature_scaler`
     3. Predict MAPE for each model using trained neural network
     4. Select model with lowest predicted MAPE

### Performance Optimization

1. **Feature Selection**:
   - Uses tsfresh's `MinimalFCParameters` for faster computation
   - Focuses on most relevant features for model selection
   - Removes redundant or highly correlated features

2. **Training Efficiency**:
   - Batch processing for faster training
   - Early stopping to prevent overfitting
   - Learning rate scheduling for better convergence

3. **Memory Management**:
   - Efficient feature extraction pipeline
   - Batch processing for large datasets
   - Memory-efficient data loading

### Model Evaluation

The classifier's performance is evaluated using:
- Mean Squared Error (MSE) on validation set
- Model selection accuracy
- MAPE prediction error
- Cross-validation scores

