#!/usr/bin/env python3
"""
MS-QCR Comprehensive Baseline Comparison
=========================================
- LSTM, GRU, TCN baselines
- Multiple metrics: MSE, MAE, RMSE, MAPE
- 5 runs with different seeds, report mean ± std
- Computational cost comparison
"""
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time

# Config
N_RUNS = 5
EPOCHS = 30
BATCH = 64
SHORT_W, LONG_W = 5, 20
N_QUBITS = 6

# Metrics
def compute_metrics(pred, actual):
    """Compute MSE, MAE, RMSE, MAPE"""
    pred = np.array(pred).flatten()
    actual = np.array(actual).flatten()
    
    mse = np.mean((pred - actual) ** 2)
    mae = np.mean(np.abs(pred - actual))
    rmse = np.sqrt(mse)
    
    # MAPE - avoid division by zero
    mask = np.abs(actual) > 1e-8
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100
    else:
        mape = 0.0
    
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# Dataset
class TimeSeriesDS(Dataset):
    def __init__(self, data, sw, lw):
        s, l, y = [], [], []
        for i in range(max(sw, lw), len(data)-1):
            s.append(data[i-sw:i]); l.append(data[i-lw:i]); y.append(data[i+1])
        self.s = np.array(s, dtype=np.float32)
        self.l = np.array(l, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32).reshape(-1, 1)
    
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return (torch.tensor(self.s[i], dtype=torch.float32).unsqueeze(0),
                torch.tensor(self.l[i], dtype=torch.float32).unsqueeze(0),
                torch.tensor(self.y[i], dtype=torch.float32))

# ============ MODELS ============

# 1. MS-QCR (Quantum-Inspired)
class MSQCR(nn.Module):
    def __init__(self):
        super().__init__()
        self.cs = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), 
                                nn.Flatten(), nn.Linear(16*SHORT_W, N_QUBITS))
        self.cl = nn.Sequential(nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2), 
                                nn.Flatten(), nn.Linear(16*LONG_W//2, N_QUBITS))
        self.fc = nn.Sequential(nn.Linear(N_QUBITS, 8), nn.ReLU(), nn.Linear(8, 1))
    
    def forward(self, xs, xl):
        f = (self.cs(xs) + self.cl(xl)) / 2
        return self.fc(f)

# 2. LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_len=25, hidden=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(nn.Linear(hidden, 8), nn.ReLU(), nn.Linear(8, 1))
    
    def forward(self, x):
        # x: (batch, seq_len)
        x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        out, (h, c) = self.lstm(x)
        return self.fc(out[:, -1, :])

# 3. GRU
class GRUModel(nn.Module):
    def __init__(self, input_len=25, hidden=32, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(1, hidden, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(nn.Linear(hidden, 8), nn.ReLU(), nn.Linear(8, 1))
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        out, h = self.gru(x)
        return self.fc(out[:, -1, :])

# 4. TCN
class TCNModel(nn.Module):
    def __init__(self, input_len=25, hidden=16):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.fc = nn.Sequential(nn.Linear(hidden * input_len, 8), nn.ReLU(), nn.Linear(8, 1))
    
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), 1, -1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ============ TRAINING ============

def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=0.01):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for e in range(epochs):
        model.train()
        for xs, xl, y in train_loader:
            # Combine windows
            x = torch.cat([xs, xl], dim=1)  # (batch, 25)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
    
    # Evaluate
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for xs, xl, y in val_loader:
            x = torch.cat([xs, xl], dim=1)
            pred = model(x)
            preds.extend(pred.squeeze().tolist())
            actuals.extend(y.squeeze().tolist())
    
    return compute_metrics(preds, actuals)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def estimate_flops(model, input_shape=(1, 25)):
    """Rough FLOPs estimate"""
    # Conv: O(kernel * channels * output_size * batch)
    # Linear: O(input * output * batch)
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            total += m.kernel_size[0] * m.in_channels * m.out_channels * 10  # approx
        elif isinstance(m, nn.Linear):
            total += m.in_features * m.out_features
    return total

# ============ MAIN ============

def run_experiment(data, data_name, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Normalize
    m, s = data.mean(), data.std() + 1e-8
    data_norm = (data - m) / s
    
    # Split
    split = int(0.8 * len(data_norm))
    train_data = data_norm[:split]
    val_data = data_norm[split:]
    
    train_ds = TimeSeriesDS(train_data, SHORT_W, LONG_W)
    val_ds = TimeSeriesDS(val_data, SHORT_W, LONG_W)
    
    train_loader = DataLoader(train_ds, BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH)
    
    results = {}
    
    # Run each model
    for model_name, ModelClass in [('MS-QCR', MSQCR), ('LSTM', LSTMModel), 
                                    ('GRU', GRUModel), ('TCN', TCNModel)]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if model_name == 'MS-QCR':
            model = MSQCR()
            # MS-QCR uses dual input
            train_losses, val_losses = [], []
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.MSELoss()
            
            for e in range(EPOCHS):
                model.train()
                for xs, xl, y in train_loader:
                    opt.zero_grad()
                    loss = criterion(model(xs, xl), y)
                    loss.backward()
                    opt.step()
            
            model.eval()
            preds, actuals = [], []
            with torch.no_grad():
                for xs, xl, y in val_loader:
                    pred = model(xs, xl)
                    preds.extend(pred.squeeze().tolist())
                    actuals.extend(y.squeeze().tolist())
        else:
            model = ModelClass(input_len=SHORT_W+LONG_W)
            train_losses, val_losses = [], []
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.MSELoss()
            
            for e in range(EPOCHS):
                model.train()
                for xs, xl, y in train_loader:
                    x = torch.cat([xs, xl], dim=1)
                    opt.zero_grad()
                    loss = criterion(model(x), y)
                    loss.backward()
                    opt.step()
            
            model.eval()
            preds, actuals = [], []
            with torch.no_grad():
                for xs, xl, y in val_loader:
                    x = torch.cat([xs, xl], dim=1)
                    pred = model(x)
                    preds.extend(pred.squeeze().tolist())
                    actuals.extend(y.squeeze().tolist())
        
        results[model_name] = compute_metrics(preds, actuals)
        results[model_name]['params'] = count_params(model)
        results[model_name]['flops'] = estimate_flops(model)
    
    return results

# ============ LOAD DATASETS ============

print("Loading datasets...")
datasets = {}

# 1. Google
df = pd.read_csv('/home/ubuntu/.openclaw/workspace/hybrid-cnn-qrc/data/GOOGLE_daily.csv')
datasets['Google'] = df['Close'].values[::-1][:500].astype(float)

# 2. BTC
df = pd.read_csv('/home/ubuntu/.openclaw/workspace/hybrid-cnn-qrc/data/btc_clean.csv')
datasets['BTC'] = df['price_num'].values[::-1][:500]

# 3. Gold
df = pd.read_csv('/home/ubuntu/.openclaw/workspace/hybrid-cnn-qrc/data/gold_stooq.csv')
datasets['Gold'] = df['Close'].values[::-1][:500]

# 4. Energy (UCI Appliances)
df = pd.read_csv('/home/ubuntu/.openclaw/workspace/hybrid-cnn-qrc/data/appliances_energy.csv')
datasets['Energy'] = df['energy'].values[:500]

# 5. Tesla
df = pd.read_csv('/home/ubuntu/.openclaw/workspace/hybrid-cnn-qrc/data/tesla_stock.csv')
datasets['Tesla'] = df['Close'].values[:500]

# 6. Apple
df = pd.read_csv('/home/ubuntu/.openclaw/workspace/hybrid-cnn-qrc/data/APPLE_daily.csv')
datasets['Apple'] = df['Close'].values[::-1][:500].astype(float)

print(f"Loaded {len(datasets)} datasets")

# ============ RUN EXPERIMENTS ============

print("\n" + "="*80)
print("RUNNING BASELINE COMPARISON (5 runs × 4 models × 6 datasets)")
print("="*80)

all_results = {name: {model: [] for model in ['MS-QCR', 'LSTM', 'GRU', 'TCN']} 
               for name in datasets.keys()}

for run in range(N_RUNS):
    print(f"\n--- Run {run+1}/{N_RUNS} ---")
    seed = 42 + run * 100
    
    for name, data in datasets.items():
        results = run_experiment(data, name, seed)
        for model_name, metrics in results.items():
            all_results[name][model_name].append(metrics)
        
        print(f"  {name}: {results['MS-QCR']['MSE']:.4f} (MS-QCR)")

# ============ AGGREGATE RESULTS ============

print("\n" + "="*80)
print("FINAL RESULTS (mean ± std over 5 runs)")
print("="*80)

print(f"\n{'Dataset':<10} | {'MS-QCR':<20} | {'LSTM':<20} | {'GRU':<20} | {'TCN':<20}")
print("-" * 100)

summary_data = []

for name in datasets.keys():
    row = [name]
    for model in ['MS-QCR', 'LSTM', 'GRU', 'TCN']:
        mses = [r['MSE'] for r in all_results[name][model]]
        mean, std = np.mean(mses), np.std(mses)
        row.append(f"{mean:.4f} ± {std:.4f}")
    
    print(f"{row[0]:<10} | {row[1]:<20} | {row[2]:<20} | {row[3]:<20} | {row[4]:<20}")
    summary_data.append(row)

# ============ COMPUTATIONAL COST ============

print("\n" + "="*80)
print("COMPUTATIONAL COST COMPARISON")
print("="*80)

# Get params for one model of each type
model_params = {}
for model_name, ModelClass in [('MS-QCR', MSQCR), ('LSTM', LSTMModel), 
                                ('GRU', GRUModel), ('TCN', TCNModel)]:
    model = ModelClass(input_len=25)
    model_params[model_name] = {
        'params': count_params(model),
        'flops': estimate_flops(model)
    }

print(f"\n{'Model':<10} | {'Parameters':<15} | {'Est. FLOPs':<15}")
print("-" * 45)
for name in ['MS-QCR', 'LSTM', 'GRU', 'TCN']:
    print(f"{name:<10} | {model_params[name]['params']:<15,} | {model_params[name]['flops']:<15,}")

# ============ SAVE RESULTS ============

import json

# Prepare detailed results
detailed = {}
for name in datasets.keys():
    detailed[name] = {}
    for model in ['MS-QCR', 'LSTM', 'GRU', 'TCN']:
        runs = all_results[name][model]
        detailed[name][model] = {
            'MSE': {'mean': np.mean([r['MSE'] for r in runs]), 'std': np.std([r['MSE'] for r in runs])},
            'MAE': {'mean': np.mean([r['MAE'] for r in runs]), 'std': np.std([r['MAE'] for r in runs])},
            'RMSE': {'mean': np.mean([r['RMSE'] for r in runs]), 'std': np.std([r['RMSE'] for r in runs])},
            'MAPE': {'mean': np.mean([r['MAPE'] for r in runs]), 'std': np.std([r['MAPE'] for r in runs])},
        }

with open('/home/ubuntu/.openclaw/workspace/hybrid-cnn-qrc/docs/baseline_results.json', 'w') as f:
    json.dump(detailed, f, indent=2)

print("\n✅ Results saved to docs/baseline_results.json")
