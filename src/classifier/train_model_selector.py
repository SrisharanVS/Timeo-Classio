import numpy as np
import pandas as pd
import torch
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

# Add the project root to Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.classifier.models import MAPEPredictor

def train_mape_predictor(
    model: nn.Module,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> tuple[list[float], list[float]]:
    """
    Train the MAPE Predictor model.
    """
    # Move model and data to device
    model = model.to(device)
    train_features = train_features.to(device)
    train_targets = train_targets.to(device)
    val_features = val_features.to(device)
    val_targets = val_targets.to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(train_features, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(val_features, val_targets)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training history
    train_losses = []
    val_losses = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.4f} - '
              f'Val Loss: {val_loss:.4f}')
        
        if patience_counter >= early_stopping_patience:
            print('Early stopping triggered')
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses

def train_model_selector(data_path: str, models_dir: str = "models", epochs: int = 100, batch_size: int = 32):
    """
    Train the model selector neural network and save all necessary files
    
    Args:
        data_path: Path to the training data CSV file
        models_dir: Directory to save models and scalers
        epochs: Number of training epochs
        batch_size: Training batch size
    """
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Load training data
    print("Loading training data...")
    data = pd.read_csv(data_path)
    
    # Separate features and MAPE values
    feature_cols = [col for col in data.columns if col.startswith('value__')]
    mape_cols = [col for col in data.columns if col.startswith('MAPE_')]
    
    X = data[feature_cols]
    y = data[mape_cols]
    
    # Handle missing and infinite values in MAPE values
    # Replace NaN with a large number (10.0 = 1000% error)
    y = y.fillna(10.0)
    # Replace infinite values with 10.0 (1000% error)
    y = y.replace([np.inf, -np.inf], 10.0)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    
    # Scale MAPE values
    mape_scaler = StandardScaler()
    y_train_scaled = mape_scaler.fit_transform(y_train)
    y_val_scaled = mape_scaler.transform(y_val)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val_scaled)
    
    # Train neural network
    print("Training neural network...")
    input_size = X_train.shape[1]
    model = MAPEPredictor(input_size=input_size)
    
    train_mape_predictor(
        model=model,
        train_features=X_train_tensor,
        train_targets=y_train_tensor,
        val_features=X_val_tensor,
        val_targets=y_val_tensor,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Save models and scalers
    print("Saving models and scalers...")
    
    # Save neural network with metadata
    torch.save({
        'state_dict': model.state_dict(),
        'feature_cols': feature_cols,
        'mape_cols': mape_cols
    }, os.path.join(models_dir, "mape_predictor.pth"))
    
    # Save scalers
    joblib.dump(feature_scaler, os.path.join(models_dir, "feature_scaler.joblib"))
    joblib.dump(mape_scaler, os.path.join(models_dir, "mape_scaler.joblib"))
    
    print("Training complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train the model selector")
    parser.add_argument("data_path", help="Path to the training data CSV file")
    parser.add_argument("--models-dir", default="models", help="Directory to save models")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    
    args = parser.parse_args()
    train_model_selector(
        data_path=args.data_path,
        models_dir=args.models_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    ) 