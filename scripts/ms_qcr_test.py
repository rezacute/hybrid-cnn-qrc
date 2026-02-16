#!/usr/bin/env python3
"""MS-QCR: Test with N_QUBITS"""
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pennylane as qml
import matplotlib.pyplot as plt
import sys

# Config - take qubit count from argument
N_QUBITS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
SHORT_W, LONG_W = 5, 20
EPOCHS = 30
BATCH = 64
N_SAMPLES = 500

print(f"=== MS-QCR with {N_QUBITS} Qubits ===")

# Data - Load Energy Demand
def make_data(n=500):
    import pandas as pd
    df = pd.read_csv('/home/ubuntu/.openclaw/workspace/hybrid-cnn-qrc/data/energy_demand.csv')
    values = df['Demand'].values.astype(np.float32)
    while len(values) < n:
        values = np.concatenate([values, values])
    return values[:n]

class DS(Dataset):
    def __init__(self, d):
        s, l, y = [], [], []
        for i in range(max(SHORT_W, LONG_W), len(d)-1):
            s.append(d[i-SHORT_W:i]); l.append(d[i-LONG_W:i]); y.append(d[i+1])
        m, s1 = d.mean(), d.std()+1e-8
        self.s = (np.array(s)-m)/s1
        self.l = (np.array(l)-m)/s1
        self.y = (np.array(y)-m)/s1
    def __len__(self): return len(self.y)
    def __getitem__(self,i): 
        return torch.tensor(self.s[i], dtype=torch.float32).unsqueeze(0), \
               torch.tensor(self.l[i], dtype=torch.float32).unsqueeze(0), \
               torch.tensor([self.y[i]], dtype=torch.float32)

# Quantum
dev = qml.device("lightning.gpu", wires=N_QUBITS)
@qml.qnode(dev)
def circuit(x, w):
    for i in range(N_QUBITS): qml.RY(np.pi*x[i], wires=i)
    qml.StronglyEntanglingLayers(w, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# Model
class MSQCR(nn.Module):
    def __init__(self):
        super().__init__()
        # Use fixed dimensions regardless of qubit count
        self.cs = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(16*SHORT_W, 8))
        self.cl = nn.Sequential(nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2), nn.Flatten(), nn.Linear(16*LONG_W//2, 8))
        
        # Quantum projection - 16 features -> N_QUBITS
        self.q_proj = nn.Linear(16, N_QUBITS)
        
        # Fixed quantum weights
        self.register_buffer('q_weights', torch.randn(3, N_QUBITS, 3))
        
        # Readout
        self.fc = nn.Linear(N_QUBITS, 1)
    
    def forward(self, xs, xl):
        fs, fl = self.cs(xs), self.cl(xl)
        f = torch.cat([fs, fl], dim=1)  # (batch, 16)
        
        # Project to quantum space
        q_in = torch.tanh(self.q_proj(f))  # (batch, N_QUBITS)
        
        return self.fc(q_in)

# Train
d = make_data(N_SAMPLES)
ds = DS(d)
tr, va = DataLoader(ds, BATCH, shuffle=True), DataLoader(ds, BATCH)

m = MSQCR()
o = torch.optim.Adam(m.parameters(), lr=0.01)
c = nn.MSELoss()

print(f"Train: {len(tr)*BATCH}, Val: {len(va)*BATCH}")

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
plt.title(f"MS-QCR Forecast ({N_QUBITS} Qubits)")
plt.legend()
plt.savefig(f'/home/ubuntu/.openclaw/workspace/hybrid-cnn-qrc/scripts/forecast_{N_QUBITS}q.png', dpi=150)
print(f"Saved: forecast_{N_QUBITS}q.png")
