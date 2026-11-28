import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

df_m = pd.read_csv('data/processed/merged_data_clean.csv', usecols=['Tanggal_dt', 'Kurs Jual', 'Kurs Beli'])
df_m['Tanggal_dt'] = pd.to_datetime(df_m['Tanggal_dt'])
df_m = df_m.sort_values('Tanggal_dt').reset_index(drop=True)
df_m['rate'] = (df_m['Kurs Jual'] + df_m['Kurs Beli']) / 2.0
df_m['log_ret'] = np.log(df_m['rate'] / df_m['rate'].shift(1))

unm = pd.read_csv('data/processed/news_processed_unmerged.csv', usecols=['aligned_trading_date'])
unm_dates = pd.to_datetime(unm['aligned_trading_date']).dt.normalize()
emb_t = np.load('data/cache_embeddings/bert_title.npy')
emb_b = np.load('data/cache_embeddings/bert_fulltext.npy')
emb_1536 = np.hstack([emb_t, emb_b])

date_to_idx = {d: i for i, d in enumerate(df_m['Tanggal_dt'])}
X_daily = np.zeros((len(df_m), 1536), dtype=np.float32)
for dt_val, grp in unm.groupby(unm_dates):
    if dt_val in date_to_idx:
        X_daily[date_to_idx[dt_val]] = emb_1536[grp.index].mean(axis=0)

X_use, y_use = X_daily[1:], df_m['log_ret'].values[1:]
n = len(y_use)
n_tr, n_va = int(n * 0.70), int(n * 0.15)
X_tr, X_va, X_te = X_use[:n_tr], X_use[n_tr:n_tr+n_va], X_use[n_tr+n_va:]
y_tr, y_va, y_te = y_use[:n_tr], y_use[n_tr:n_tr+n_va], y_use[n_tr+n_va:]

mu_y, std_y = y_tr.mean(), y_tr.std()
y_tr_s, y_va_s = (y_tr - mu_y) / std_y, (y_va - mu_y) / std_y
sc = StandardScaler().fit(X_tr)
pca = PCA(n_components=32, random_state=SEED).fit(sc.transform(X_tr))
Z_tr, Z_va, Z_te = pca.transform(sc.transform(X_tr)), pca.transform(sc.transform(X_va)), pca.transform(sc.transform(X_te))

class CompactMLP(nn.Module):
    def __init__(self, in_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16), nn.Tanh(), nn.Dropout(0.35),
            nn.Linear(16, 8), nn.Tanh(),
            nn.Linear(8, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

for dw in [0.0, 0.2]:
    torch.manual_seed(SEED)
    m = CompactMLP(32)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.05)
    Xt, Xv, Xte_t = torch.tensor(Z_tr, dtype=torch.float32), torch.tensor(Z_va, dtype=torch.float32), torch.tensor(Z_te, dtype=torch.float32)
    yt, yv = torch.tensor(y_tr_s, dtype=torch.float32), torch.tensor(y_va_s, dtype=torch.float32)
    st = torch.tensor(np.where(y_tr > 0, 1.0, -1.0), dtype=torch.float32)
    sv = torch.tensor(np.where(y_va > 0, 1.0, -1.0), dtype=torch.float32)
    best_v, best_w = float('inf'), None
    for ep in range(120):
        m.train(); opt.zero_grad()
        ps = m(Xt)
        loss = torch.mean((ps - yt)**2) + dw * torch.mean(torch.nn.functional.softplus(-st * (ps + mu_y/std_y) * 3.0))
        loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pv = m(Xv)
            vl = torch.mean((pv - yv)**2).item() + dw * torch.mean(torch.nn.functional.softplus(-sv * (pv + mu_y/std_y) * 3.0)).item()
        if vl < best_v: best_v = vl; best_w = copy.deepcopy(m.state_dict())
    m.load_state_dict(best_w); m.eval()
    with torch.no_grad(): yp = m(Xte_t).numpy() * std_y + mu_y
    err = y_te - yp
    rmse, mae = np.sqrt(np.mean(err**2)), np.mean(np.abs(err))
    v = (y_te != 0) & (yp != 0)
    da = np.mean(np.sign(y_te[v]) == np.sign(yp[v])) * 100
    ba = balanced_accuracy_score((y_te > 0).astype(int), (yp > 0).astype(int)) * 100
    print(f"CompactMLP PCA-32 (dw={dw}): RMSE={rmse:.6f} MAE={mae:.6f} DirAcc={da:.2f}% BalAcc={ba:.2f}%")
