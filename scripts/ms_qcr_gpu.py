#!/usr/bin/env python3
"""
MS-QCR Optimized for L40S GPU
Multi-Scale Quantum-CNN Reservoir with Batch Processing
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pennylane as qml
from pennylane.templates import StronglyEntanglingLayers
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
class CFG:
    SHORT_WINDOW = 5
    LONG_WINDOW = 20
    N_SHORT = 2
    N_LONG = 2
    N_BRIDGE = 1
    N_QUBITS = N_SHORT + N_LONG + N_BRIDGE  # 5
    EPOCHS = 50
    BATCH_SIZE = 64  # Larger batch
    LR = 0.01
    TRAIN_SPLIT = 0.8
    N_SAMPLES = 2000

cfg = CFG()

# ============================================================
# DATASET
# ============================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, data, short_w=5, long_w=20):
        self.short, self.long, self.y = [], [], []
        for i in range(max(short_w, long_w), len(data)-1):
            self.short.append(data[i-short_w:i])
            self.long.append(data[i-long_w:i])
            self.y.append(data[i+1])
        
        self.short = np.array(self.short, dtype=np.float32)
        self.long = np.array(self.long, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)
        
        # Normalize
        for x in [self.short, self.long, self.y]:
            m, s = x.mean(), x.std() + 1e-8
            x -= m; x /= s
    
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return (torch.FloatTensor(self.short[i]).unsqueeze(0),
                torch.FloatTensor(self.long[i]).unsqueeze(0),
                torch.FloatTensor([self.y[i]]))

def generate_data(n=2000):
    t = np.linspace(0, 100, n)
    return (np.sin(0.5*t) + 0.5*np.sin(0.05*t) + np.random.randn(n)*0.1).astype(np.float32)

# ============================================================
# GPU QUANTUM DEVICE
# ============================================================
# Use lightning.gpu for L40S
dev = qml.device("lightning.gpu", wires=cfg.N_QUBITS)

@qml.qnode(dev, diff_method="adjoint")
def quantum_circuit(inputs, weights):
    """Vectorized quantum circuit"""
    # Encoding - use angles proportional to inputs
    for i in range(cfg.N_QUBITS):
        qml.RY(np.pi * inputs[i], wires=i)
    
    # Entanglement - strongly entangling layers
    StronglyEntanglingLayers(weights, wires=range(cfg.N_QUBITS))
    
    return [qml.expval(qml.PauliZ(i)) for i in range(cfg.N_QUBITS)]

# ============================================================
# MODEL - OPTIMIZED
# ============================================================
class MS_QCR_Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        # CNN Encoders
        self.net_short = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8*cfg.SHORT_WINDOW, cfg.N_SHORT)
        )
        
        self.net_long = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(8*(cfg.LONG_WINDOW//2), cfg.N_LONG)
        )
        
        # Quantum Reservoir - fixed weights
        self.register_buffer('reservoir_weights', 
            torch.randn(3, cfg.N_QUBITS, 3))  # shape for StronglyEntanglingLayers
        
        # Readout
        self.readout = nn.Linear(cfg.N_QUBITS, 1)
    
    def forward(self, x_short, x_long):
        # CNN encoding
        f_s = self.net_short(x_short)
        f_l = self.net_long(x_long)
        combined = torch.cat([f_s, f_l], dim=1)
        
        # Bridge = mean
        bridge = combined.mean(dim=1, keepdim=True)
        quantum_in = torch.cat([combined, bridge], dim=1)  # (batch, 5)
        
        # Quantum - batch process
        q_out = torch.zeros(quantum_in.shape[0], cfg.N_QUBITS, device=quantum_in.device)
        weights = self.reservoir_weights
        
        for i in range(quantum_in.shape[0]):
            inputs = quantum_in[i].detach().cpu().numpy()
            w = weights.detach().cpu().numpy()
            result = quantum_circuit(inputs, w)
            q_out[i] = torch.tensor([float(v) for v in result], device=quantum_in.device)
        
        return self.readout(q_out)

# ============================================================
# TRAINING
# ============================================================
def train():
    print("="*50)
    print("MS-QCR GPU Optimized")
    print("="*50)
    
    # Data
    data = generate_data(cfg.N_SAMPLES)
    ds = TimeSeriesDataset(data, cfg.SHORT_WINDOW, cfg.LONG_WINDOW)
    n_train = int(len(ds) * cfg.TRAIN_SPLIT)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, len(ds)-n_train])
    
    train_loader = DataLoader(train_ds, cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, cfg.BATCH_SIZE)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model
    model = MS_QCR_Model()
    print(f"Params: {sum(p.numel() for p in model.parameters())}")
    
    # Train
    opt = optim.Adam(model.parameters(), lr=cfg.LR)
    criterion = nn.MSELoss()
    
    for epoch in range(cfg.EPOCHS):
        model.train()
        train_loss = 0
        for x_s, x_l, y in train_loader:
            opt.zero_grad()
            pred = model(x_s, x_l)
            loss = criterion(pred, y)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        
        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_s, x_l, y in val_loader:
                pred = model(x_s, x_l)
                val_loss += criterion(pred, y).item()
        
        print(f"Epoch {epoch+1:2d} | Train: {train_loss/len(train_loader):.4f} | Val: {val_loss/len(val_loader):.4f}")
    
    # Plot
    model.eval()
    preds, acts = [], []
    with torch.no_grad():
        for x_s, x_l, y in val_loader:
            pred = model(x_s, x_l)
            preds.extend(pred.squeeze().tolist())
            acts.extend(y.squeeze().tolist())
    
    plt.figure(figsize=(12,5))
    plt.plot(acts, label='Actual')
    plt.plot(preds, label='Predicted')
    plt.legend()
    plt.title("MS-QCR Forecast")
    plt.savefig('forecast.png', dpi=150)
    print("\nSaved: forecast.png")

if __name__ == "__main__":
    train()
