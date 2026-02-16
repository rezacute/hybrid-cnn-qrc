#!/usr/bin/env python3
"""MS-QCR: Ultra-Fast Classical Baseline"""
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Config
SHORT_W, LONG_W = 5, 20
EPOCHS = 30
BATCH = 64

# Data - Load real dataset
def make_data(n=2000):
    """Load Tesla Stock dataset"""
    import pandas as pd
    
    # Load real dataset
    df = pd.read_csv('/home/ubuntu/.openclaw/workspace/hybrid-cnn-qrc/data/tesla_stock.csv')
    print(f"Tesla data columns: {df.columns.tolist()}")
    
    # Try different column names
    for col in ['Close', 'close', 'Adj Close', 'adj_close', 'price']:
        if col in df.columns:
            values = df[col].values.astype(np.float32)
            break
    else:
        # Use first numeric column
        values = df.iloc[:, 4].values.astype(np.float32)
    
    print(f"Tesla stock: {len(values)} points")
    
    # Repeat to match desired length
    while len(values) < n:
        values = np.concatenate([values, values])
    
    return values[:n]

class DS(Dataset):
    def __init__(self, d):
        s, l, y = [], [], []
        for i in range(max(SHORT_W, LONG_W), len(d)-1):
            s.append(d[i-SHORT_W:i]); l.append(d[i-LONG_W:i]); y.append(d[i+1])
        m, s1 = np.mean(s), np.std(s)+1e-8
        m, s2 = np.mean(l), np.std(l)+1e-8
        m, s3 = np.mean(y), np.std(y)+1e-8
        self.s = (np.array(s)-m)/s1
        self.l = (np.array(l)-m)/s2
        self.y = (np.array(y)-m)/s3
    def __len__(self): return len(self.y)
    def __getitem__(self,i): 
        return torch.tensor(self.s[i], dtype=torch.float32).unsqueeze(0), \
               torch.tensor(self.l[i], dtype=torch.float32).unsqueeze(0), \
               torch.tensor([self.y[i]], dtype=torch.float32)

# Model - Pure Classical CNN
class MSQCR(nn.Module):
    def __init__(self):
        super().__init__()
        self.cs = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(16*SHORT_W, 8))
        self.cl = nn.Sequential(nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2), nn.Flatten(), nn.Linear(16*LONG_W//2, 8))
        self.fc = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
    
    def forward(self, xs, xl):
        fs, fl = self.cs(xs), self.cl(xl)
        f = torch.cat([fs, fl], dim=1)
        return self.fc(f)

# Train
print("Generating data...")
d = make_data(2000)
ds = DS(d)
tr, va = DataLoader(ds, BATCH, shuffle=True), DataLoader(ds, BATCH)
print(f"Train: {len(tr)*BATCH}, Val: {len(va)*BATCH}")

m = MSQCR()
o = torch.optim.Adam(m.parameters(), lr=0.01)
c = nn.MSELoss()

print("Training...")
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
plt.title("MS-QCR Forecast (Classical)")
plt.legend()
plt.savefig('/home/ubuntu/.openclaw/workspace/hybrid-cnn-qrc/scripts/forecast.png', dpi=150)
print("Saved: forecast.png")
