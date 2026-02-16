#!/usr/bin/env python3
"""
Multi-Scale Quantum-CNN Reservoir (MS-QCR)
Time Series Forecasting with Hybrid Classical-Quantum Architecture
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pennylane as qml
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    # Window sizes
    SHORT_WINDOW = 5   # High-frequency capture
    LONG_WINDOW = 20    # Trend/seasonality capture
    
    # Quantum reservoir config
    N_SHORT = 2        # Zone 1: Short features
    N_LONG = 2         # Zone 2: Long features  
    N_BRIDGE = 1      # Zone 3: Bridge qubit
    N_QUBITS = N_SHORT + N_LONG + N_BRIDGE  # Total: 5
    
    # Training
    EPOCHS = 50
    BATCH_SIZE = 16
    LEARNING_RATE = 0.01
    TRAIN_SPLIT = 0.8
    
    # Data
    N_SAMPLES = 1000
    NOISE_LEVEL = 0.1

cfg = Config()

# ============================================================
# DATASET: DUAL-WINDOW SLICING
# ============================================================
class TimeSeriesDataset(Dataset):
    """Dataset that creates short and long window inputs for each time step"""
    
    def __init__(self, data, short_window=5, long_window=20):
        self.data = data
        self.short_window = short_window
        self.long_window = long_window
        
        # Create windows for each target point
        self.X_short = []
        self.X_long = []
        self.y = []
        
        for i in range(max(short_window, long_window), len(data) - 1):
            # Short window: last 5 points
            self.X_short.append(data[i - short_window:i])
            # Long window: last 20 points
            self.X_long.append(data[i - long_window:i])
            # Target: next point
            self.y.append(data[i + 1])
        
        self.X_short = np.array(self.X_short, dtype=np.float32)
        self.X_long = np.array(self.X_long, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)
        
        # Normalize
        self.X_short = (self.X_short - self.X_short.mean()) / (self.X_short.std() + 1e-8)
        self.X_long = (self.X_long - self.X_long.mean()) / (self.X_long.std() + 1e-8)
        self.y = (self.y - self.y.mean()) / (self.y.std() + 1e-8)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # Add channel dimension for Conv1d
        x_short = torch.FloatTensor(self.X_short[idx]).unsqueeze(0)  # (1, 5)
        x_long = torch.FloatTensor(self.X_long[idx]).unsqueeze(0)    # (1, 20)
        y = torch.FloatTensor([self.y[idx]])
        return x_short, x_long, y


# ============================================================
# DATA GENERATION: SUM OF SINES
# ============================================================
def generate_synthetic_data(n_samples=1000, noise_level=0.1):
    """Generate synthetic time series: sum of fast + slow sine waves"""
    t = np.linspace(0, 50, n_samples)
    
    # Fast oscillation (high frequency)
    fast = np.sin(0.5 * t)
    # Slow oscillation (low frequency)
    slow = 0.5 * np.sin(0.05 * t)
    
    # Combine + noise
    data = fast + slow + np.random.randn(n_samples) * noise_level
    
    return data.astype(np.float32)


# ============================================================
# QUANTUM RESERVOIR CIRCUIT
# ============================================================
# Quantum device - use lightning for speed
try:
    dev = qml.device("lightning.gpu", wires=5)
except:
    dev = qml.device("default.qubit", wires=5)

@qml.qnode(dev)
def quantum_reservoir_circuit(inputs, reservoir_weights):
    """Quantum Reservoir with fixed (non-trainable) weights"""
    # Encoding
    qml.RY(inputs[0], wires=0)
    qml.RY(inputs[1], wires=1)
    qml.RY(inputs[3], wires=3)
    qml.RY(inputs[4], wires=4)
    
    # Entanglement
    weights = reservoir_weights.reshape(3, 5, 5)
    for layer in range(3):
        for i in range(5):
            qml.RX(weights[layer, i, 0], wires=i)
            qml.RY(weights[layer, i, 1], wires=i)
        for i in range(5):
            for j in range(i + 1, 5):
                qml.CNOT(wires=[i, j])
                qml.RZ(weights[layer, i, j], wires=j)
                qml.CNOT(wires=[i, j])
    
    return [qml.expval(qml.PauliZ(i)) for i in range(5)]


# ============================================================
# MS-QCR MODEL
# ============================================================
class MS_QCR_Model(nn.Module):
    """
    Multi-Scale Quantum-CNN Reservoir Model
    
    Architecture:
    - Stage A: Dual-window inputs (handled by Dataset)
    - Stage B: Parallel CNN encoders (trainable)
    - Stage C: Quantum Reservoir (fixed weights)
    - Stage D: Readout layer (trainable)
    """
    
    def __init__(self):
        super().__init__()
        
        # ---- STAGE B: PARALLEL CLASSICAL ENCODERS ----
        
        # Short Encoder: Conv1d(1, 4, kernel=3) -> Flatten -> Linear -> 2
        self.net_short = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * cfg.SHORT_WINDOW, cfg.N_SHORT)  # Output: 2
        )
        
        # Long Encoder: Conv1d(1, 4, kernel=5) -> Flatten -> Linear -> 2
        # Input: (1, 20) -> Conv1d(1, 4, 5) -> (4, 16) -> Pool -> (4, 8) -> Flatten -> Linear -> 2
        self.net_long = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(4 * (cfg.LONG_WINDOW // 2), cfg.N_LONG)  # Output: 2
        )
        
        # ---- STAGE C: QUANTUM RESERVOIR (FIXED WEIGHTS) ----
        # Initialize random reservoir weights
        # Shape: (n_layers * n_qubits * n_qubits) = 3 * 5 * 5 = 75
        n_layers = 3
        reservoir_params = np.random.randn(n_layers, cfg.N_QUBITS, cfg.N_QUBITS).astype(np.float32)
        self.reservoir_weights = nn.Parameter(
            torch.tensor(reservoir_params), 
            requires_grad=False  # CRITICAL: Fixed weights = Reservoir effect
        )
        
        # ---- STAGE D: READOUT LAYER ----
        self.readout = nn.Linear(cfg.N_QUBITS, 1)
    
    def forward(self, x_short, x_long):
        """
        Forward pass
        
        Args:
            x_short: Short window input (batch, 1, 5)
            x_long: Long window input (batch, 1, 20)
            
        Returns:
            predictions: (batch, 1)
        """
        # ---- Stage B: CNN Encoding ----
        feat_short = self.net_short(x_short)  # (batch, 2)
        feat_long = self.net_long(x_long)    # (batch, 2)
        
        # Combine short and long features (4 total)
        combined = torch.cat([feat_short, feat_long], dim=1)  # (batch, 4)
        
        # ---- Stage C: Quantum Reservoir ----
        # Process each sample in batch (convert to numpy for speed)
        quantum_outputs = []
        reservoir_w = self.reservoir_weights.detach().numpy()
        for i in range(combined.shape[0]):
            q_input = np.array([
                combined[i, 0].item(),
                combined[i, 1].item(),
                0.0,
                combined[i, 2].item(),
                combined[i, 3].item(),
            ], dtype=np.float64)
            
            q_out = quantum_reservoir_circuit(q_input, reservoir_w)
            q_out = torch.tensor([float(v) for v in q_out], dtype=torch.float32)
            quantum_outputs.append(q_out)
        
        quantum_outputs = torch.stack(quantum_outputs, dim=0)  # (batch, 5)
        
        # ---- Stage D: Readout ----
        prediction = self.readout(quantum_outputs)  # (batch, 1)
        
        return prediction


# ============================================================
# TRAINING
# ============================================================
def train_model(model, train_loader, val_loader, epochs=50, lr=0.01):
    """Train the MS-QCR model"""
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for x_short, x_long, y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(x_short, x_long)
            loss = criterion(pred, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_short, x_long, y in val_loader:
                pred = model(x_short, x_long)
                loss = criterion(pred, y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    return train_losses, val_losses


# ============================================================
# VISUALIZATION
# ============================================================
def plot_predictions(model, dataset, split_idx, title="MS-QCR Forecast"):
    """Plot predictions vs actuals"""
    model.eval()
    
    # Get all data
    all_preds = []
    all_actuals = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            x_short, x_long, y = dataset[i]
            x_short = x_short.unsqueeze(0)
            x_long = x_long.unsqueeze(0)
            
            pred = model(x_short, x_long)
            all_preds.append(pred.item())
            all_actuals.append(y.item())
    
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    
    # Denormalize (approximate)
    # For visualization, we show normalized values
    
    plt.figure(figsize=(14, 6))
    
    # Plot training portion
    plt.plot(range(split_idx), all_actuals[:split_idx], 'b-', label='Actual (Train)', alpha=0.7)
    plt.plot(range(split_idx), all_preds[:split_idx], 'g--', label='Pred (Train)', alpha=0.7)
    
    # Plot validation portion
    plt.plot(range(split_idx, len(all_actuals)), all_actuals[split_idx:], 'b-', label='Actual (Val)', linewidth=2)
    plt.plot(range(split_idx, len(all_preds)), all_preds[split_idx:], 'r-', label='Pred (Val)', linewidth=2)
    
    plt.axvline(x=split_idx, color='k', linestyle='--', label='Train/Val Split')
    
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('forecast_results.png', dpi=150)
    plt.show()
    
    # Calculate metrics
    train_rmse = np.sqrt(np.mean((all_preds[:split_idx] - all_actuals[:split_idx])**2))
    val_rmse = np.sqrt(np.mean((all_preds[split_idx:] - all_actuals[split_idx:])**2))
    
    print(f"\nFinal Results:")
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Val RMSE: {val_rmse:.4f}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("Multi-Scale Quantum-CNN Reservoir (MS-QCR)")
    print("=" * 60)
    
    # Generate data
    print("\n[1] Generating synthetic time series...")
    data = generate_synthetic_data(n_samples=cfg.N_SAMPLES, noise_level=cfg.NOISE_LEVEL)
    
    # Create dataset
    print("[2] Creating dataset with dual-window slicing...")
    dataset = TimeSeriesDataset(
        data, 
        short_window=cfg.SHORT_WINDOW, 
        long_window=cfg.LONG_WINDOW
    )
    
    # Split
    split_idx = int(len(dataset) * cfg.TRAIN_SPLIT)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [split_idx, len(dataset) - split_idx]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE)
    
    print(f"    Train samples: {len(train_dataset)}")
    print(f"    Val samples: {len(val_dataset)}")
    
    # Initialize model
    print("\n[3] Initializing MS-QCR Model...")
    model = MS_QCR_Model()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"    Total parameters: {total_params}")
    print(f"    Trainable (CNN + Readout): {trainable_params}")
    print(f"    Frozen (Quantum Reservoir): {frozen_params}")
    
    # Train
    print("\n[4] Training...")
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=cfg.EPOCHS, 
        lr=cfg.LEARNING_RATE
    )
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Plot
    print("\n[5] Generating forecast plot...")
    plot_predictions(model, dataset, split_idx)
    
    print("\n" + "=" * 60)
    print("Done! Check 'forecast_results.png'")
    print("=" * 60)


if __name__ == "__main__":
    main()
