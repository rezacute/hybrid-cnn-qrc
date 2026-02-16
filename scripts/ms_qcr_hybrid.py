#!/usr/bin/env python3
"""MS-QCR: Hybrid Quantum with Learnable Parameters"""
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pennylane as qml
import matplotlib.pyplot as plt

# Config
N_QUBITS = 6
SHORT_W, LONG_W = 5, 20
EPOCHS = 50
BATCH = 32
N_SAMPLES = 300
N_LAYERS = 2

print(f"=== LEARNABLE QUANTUM: {N_QUBITS} Qubits ===")

# Data
def make_data(n=300):
    import pandas as pd
    df = pd.read_csv('/home/ubuntu/.openclaw/workspace/hybrid-cnn-qrc/data/energy_demand.csv')
    return df['Demand'].values.astype(np.float32)[:n]

class DS(Dataset):
    def __init__(self, d):
        s, l, y = [], [], []
        for i in range(max(SHORT_W, LONG_W), len(d)-1):
            s.append(d[i-SHORT_W:i]); l.append(d[i-LONG_W:i]); y.append(d[i+1])
        m, s_std = np.mean(d), np.std(d)+1e-8
        self.s = (np.array(s)-m)/s_std
        self.l = (np.array(l)-m)/s_std
        self.y = (np.array(y)-m)/s_std
    def __len__(self): return len(self.y)
    def __getitem__(self,i): 
        return (torch.tensor(self.s[i], dtype=torch.float32).unsqueeze(0),
                torch.tensor(self.l[i], dtype=torch.float32).unsqueeze(0),
                torch.tensor([self.y[i]], dtype=torch.float32))

# Quantum
dev = qml.device("lightning.gpu", wires=N_QUBITS)

class QuantumLayer(nn.Module):
    """Differentiable quantum layer"""
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # Learnable weights
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
    
    def forward(self, x):
        """Execute quantum circuit for batch"""
        batch_size = x.shape[0]
        outputs = []
        
        x_detach = x.detach()
        
        for i in range(batch_size):
            inputs = x_detach[i].cpu().numpy()
            
            @qml.qnode(dev, diff_method="adjoint")
            def circuit(inp, w):
                # Angle embedding
                for j in range(self.n_qubits):
                    qml.RY(np.pi * inp[j], wires=j)
                # Entanglement
                qml.StronglyEntanglingLayers(w, wires=range(self.n_qubits))
                return [qml.expval(qml.PauliZ(k)) for k in range(self.n_qubits)]
            
            out = circuit(inputs, self.weights.detach().cpu().numpy())
            outputs.append(out)
        
        return torch.tensor(outputs, dtype=torch.float32, device=x.device)

# Model with Quantum Layer
class HybridQRC(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN encoders
        self.cs = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), 
                               nn.Flatten(), nn.Linear(16*SHORT_W, N_QUBITS))
        self.cl = nn.Sequential(nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                               nn.Flatten(), nn.Linear(16*LONG_W//2, N_QUBITS))
        
        # Quantum layer
        self.quantum = QuantumLayer(N_QUBITS, N_LAYERS)
        
        # Classical post-processing
        self.fc = nn.Sequential(nn.Linear(N_QUBITS, 8), nn.ReLU(), nn.Linear(8, 1))
    
    def forward(self, xs, xl):
        # CNN features
        fs = self.cs(xs)
        fl = self.cl(xl)
        
        # Combine and project
        f = (fs + fl) / 2  # Average pooling
        
        # Quantum transformation
        q_out = self.quantum(f)
        
        # Classical readout
        return self.fc(q_out)

# Train
d = make_data(N_SAMPLES)
ds = DS(d)
tr = DataLoader(ds, BATCH, shuffle=True)
va = DataLoader(ds, BATCH)

model = HybridQRC()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print(f"Samples: {len(ds)}, Batches: {len(tr)}")

for e in range(EPOCHS):
    model.train()
    tl = 0
    for xs, xl, y in tr:
        opt.zero_grad()
        pred = model(xs, xl)
        loss = criterion(pred, y)
        loss.backward()
        opt.step()
        tl += loss.item()
    
    model.eval()
    vl = sum(criterion(model(xs, xl), y).item() for xs, xl, y in va) / len(va)
    
    if e % 10 == 0:
        print(f"E{e+1:2d} | Train: {tl/len(tr):.4f} | Val: {vl:.4f}")

# Final
model.eval()
vl = sum(criterion(model(xs, xl), y).item() for xs, xl, y in va) / len(va)
print(f"\n=== HYBRID QUANTUM Val MSE: {vl:.4f} ===")

# Plot
ps, at = [], []
for xs, xl, y in va:
    ps.extend(model(xs, xl).squeeze().tolist())
    at.extend(y.squeeze().tolist())

plt.figure(figsize=(10,4))
plt.plot(at, label='Actual')
plt.plot(ps, label='Predicted')
plt.title(f"Hybrid Quantum ({N_QUBITS}Q)")
plt.legend()
plt.savefig('/home/ubuntu/.openclaw/workspace/hybrid-cnn-qrc/scripts/forecast_hybrid_quantum.png', dpi=150)
print("Saved: forecast_hybrid_quantum.png")
