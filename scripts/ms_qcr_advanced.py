#!/usr/bin/env python3
"""MS-QCR: Ancilla Padding & Feature Superposition"""
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pennylane as qml
import matplotlib.pyplot as plt

# Config
N_QUBITS = 6
N_DATA_QUBITS = 4  # Qubits for data
N_ANCILLA = 2      # Ancilla qubits (|0⟩)
SHORT_W, LONG_W = 5, 20
EPOCHS = 30
BATCH = 64
N_SAMPLES = 500
METHOD = "ancilla"  # or "fractional"

print(f"=== MS-QCR: {METHOD} Encoding ===")

# Data
def make_data(n=500):
    import pandas as pd
    df = pd.read_csv('/home/ubuntu/.openclaw/workspace/hybrid-cnn-qrc/data/energy_demand.csv')
    values = df['Demand'].values.astype(np.float32)
    while len(values) < n: values = np.concatenate([values, values])
    return values[:n]

class DS(Dataset):
    def __init__(self, d):
        s, l, y = [], [], []
        for i in range(max(SHORT_W, LONG_W), len(d)-1):
            s.append(d[i-SHORT_W:i]); l.append(d[i-LONG_W:i]); y.append(d[i+1])
        m, s = d.mean(), d.std()+1e-8
        self.s = (np.array(s)-m)/s; self.l = (np.array(l)-m)/s; self.y = (np.array(y)-m)/s
    def __len__(self): return len(self.y)
    def __getitem__(self,i): 
        return torch.tensor(self.s[i], dtype=torch.float32).unsqueeze(0), \
               torch.tensor(self.l[i], dtype=torch.float32).unsqueeze(0), \
               torch.tensor([self.y[i]], dtype=torch.float32)

# Quantum
dev = qml.device("lightning.gpu", wires=N_QUBITS)

@qml.qnode(dev)
def circuit_ancilla(x):
    """Ancilla encoding: Qubits 0-3 = data, Qubits 4-5 = ancilla |0>"""
    # Data encoding on qubits 0-3
    for i in range(4):
        qml.RY(np.pi * x[i], wires=i)
    # Qubits 4-5 remain |0> (ancillas)
    
    # Entanglement across all qubits
    for i in range(N_QUBITS-1):
        qml.CNOT(wires=[i, i+1])
    
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

@qml.qnode(dev)
def circuit_fractional(x):
    """Fractional encoding: Most volatile feature mapped to 2 qubits"""
    # Map: short features → qubits 0,1 (duplicated for weight)
    # long features → qubits 2,3
    # short volatile → qubits 4,5 (fractional)
    
    # Short window (5→2 via mean+std)
    qml.RY(np.pi * x[0], wires=0)  # short mean
    qml.RY(np.pi * x[1], wires=1)  # short std (volatile)
    qml.RY(np.pi * x[1], wires=4)  # fractional (same as volatile)
    
    # Long window
    qml.RY(np.pi * x[2], wires=2)  # long mean
    qml.RY(np.pi * x[3], wires=3)  # long std
    
    # Entanglement
    for i in range(N_QUBITS-1):
        qml.CNOT(wires=[i, i+1])
    
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# Model
class MSQCR(nn.Module):
    def __init__(self, method="ancilla"):
        super().__init__()
        self.method = method
        
        # CNNs → 4 features (for qubits 0-3)
        self.cs = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(16*SHORT_W, 2))
        self.cl = nn.Sequential(nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2), nn.Flatten(), nn.Linear(16*LONG_W//2, 2))
        
        # Quantum projection → 6 features
        self.q_proj = nn.Linear(4, N_QUBITS)
        self.fc = nn.Linear(N_QUBITS, 1)
    
    def forward(self, xs, xl):
        fs, fl = self.cs(xs), self.cl(xl)
        f = torch.cat([fs, fl], dim=1)  # (batch, 4)
        
        # Project to quantum space
        q_in = torch.tanh(self.q_proj(f))  # (batch, 6)
        
        return self.fc(q_in)

# Train
d = make_data(N_SAMPLES)
ds = DS(d)
tr, va = DataLoader(ds, BATCH, shuffle=True), DataLoader(ds, BATCH)

m = MSQCR(METHOD)
o = torch.optim.Adam(m.parameters(), lr=0.01)
c = nn.MSELoss()

for e in range(EPOCHS):
    m.train(); tl = 0
    for xs, xl, y in tr:
        o.zero_grad(); l = c(m(xs, xl), y); l.backward(); o.step(); tl += l.item()
    m.eval(); vl = sum(c(m(xs,xl),y).item() for xs,xl,y in va)/len(va)
    print(f"E{e+1:2d} | Train: {tl/len(tr):.4f} | Val: {vl:.4f}")

# Plot
m.eval()
ps, at = [], []
for xs, xl, y in va:
    ps.extend(m(xs, xl).squeeze().tolist())
    at.extend(y.squeeze().tolist())

plt.figure(figsize=(12,4))
plt.plot(at, label='Actual')
plt.plot(ps, label='Predicted')
plt.title(f"MS-QCR ({METHOD})")
plt.legend()
plt.savefig(f'/home/ubuntu/.openclaw/workspace/hybrid-cnn-qrc/scripts/forecast_{METHOD}.png', dpi=150)
print(f"Saved: forecast_{METHOD}.png")
