import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import balanced_accuracy_score

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Load rate & unmerged alignment (to test 1536-d concat embeddings locally!)
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

# Test both same-day (t -> t) and lagged (t-1 -> t, like trial1 t -> t+1)
for mode in ["same_day(t)", "next_day(t-1 -> t)"]:
    print(f"\n================ MODE: {mode} ================")
    if mode == "same_day(t)":
        X_use = X_daily[1:]
        y_use = df_m['log_ret'].values[1:]
    else:
        X_use = X_daily[:-1]
        y_use = df_m['log_ret'].values[1:]

    n = len(y_use)
    n_tr, n_va = int(n * 0.70), int(n * 0.15)
    X_tr, X_va, X_te = X_use[:n_tr], X_use[n_tr:n_tr+n_va], X_use[n_tr+n_va:]
    y_tr, y_va, y_te = y_use[:n_tr], y_use[n_tr:n_tr+n_va], y_use[n_tr+n_va:]

    mu_y, std_y = y_tr.mean(), y_tr.std()
    y_tr_s = (y_tr - mu_y) / std_y
    y_va_s = (y_va - mu_y) / std_y

    sc = StandardScaler().fit(X_tr)
    X_tr_s, X_va_s, X_te_s = sc.transform(X_tr), sc.transform(X_va), sc.transform(X_te)

    def eval_m(name, yp):
        err = y_te - yp
        rmse = np.sqrt(np.mean(err**2))
        mae = np.mean(np.abs(err))
        mse = np.mean(err**2)
        v = (y_te != 0) & (yp != 0)
        da = np.mean(np.sign(y_te[v]) == np.sign(yp[v])) * 100
        ba = balanced_accuracy_score((y_te > 0).astype(int), (yp > 0).astype(int)) * 100
        print(f"  {name:32s}: RMSE={rmse:.6f} MAE={mae:.6f} MSE={mse:.8f} DirAcc={da:.2f}% BalAcc={ba:.2f}%")

    eval_m("Baseline Mean", np.full_like(y_te, mu_y))

    # 1. Ridge direct 1536-d (heavy alpha since d=1536 > n=840!)
    for a in [5000.0, 20000.0]:
        r = Ridge(alpha=a, random_state=SEED).fit(X_tr_s, y_tr_s)
        eval_m(f"Ridge 1536-d (alpha={int(a)})", r.predict(X_te_s) * std_y + mu_y)

    # 2. PCA-32 + Ridge / GBDT / RF
    pca = PCA(n_components=32, random_state=SEED).fit(X_tr_s)
    Z_tr, Z_va, Z_te = pca.transform(X_tr_s), pca.transform(X_va_s), pca.transform(X_te_s)
    r_p = Ridge(alpha=500.0, random_state=SEED).fit(Z_tr, y_tr_s)
    eval_m("Ridge PCA-32 (alpha=500)", r_p.predict(Z_te) * std_y + mu_y)

    rf = RandomForestRegressor(n_estimators=150, max_depth=5, min_samples_leaf=10, random_state=SEED, n_jobs=-1).fit(Z_tr, y_tr_s)
    eval_m("RandomForest PCA-32", rf.predict(Z_te) * std_y + mu_y)

    gbr = GradientBoostingRegressor(n_estimators=60, max_depth=2, learning_rate=0.03, subsample=0.8, random_state=SEED).fit(Z_tr, y_tr_s)
    eval_m("GBDT PCA-32", gbr.predict(Z_te) * std_y + mu_y)

    # 3. Direct PyTorch MLP on 1536-d (with strong Dropout + WeightDecay + Early Stopping!)
    class DirectEmbMLP(nn.Module):
        def __init__(self, in_dim=1536):
            super().__init__()
            self.net = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(0.4),
                nn.Linear(64, 16),
                nn.GELU(),
                nn.Linear(16, 1)
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    torch.manual_seed(SEED)
    mlp = DirectEmbMLP(1536)
    opt = torch.optim.AdamW(mlp.parameters(), lr=5e-4, weight_decay=0.1)
    Xt, Xv, Xte_t = torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(X_va_s, dtype=torch.float32), torch.tensor(X_te_s, dtype=torch.float32)
    yt, yv = torch.tensor(y_tr_s, dtype=torch.float32), torch.tensor(y_va_s, dtype=torch.float32)

    best_v, best_w = float('inf'), None
    for ep in range(80):
        mlp.train(); opt.zero_grad()
        loss = torch.mean((mlp(Xt) - yt)**2)
        loss.backward(); opt.step()
        mlp.eval()
        with torch.no_grad():
            vl = torch.mean((mlp(Xv) - yv)**2).item()
        if vl < best_v:
            best_v = vl; best_w = copy.deepcopy(mlp.state_dict())
    mlp.load_state_dict(best_w); mlp.eval()
    with torch.no_grad():
        p_mlp = mlp(Xte_t).numpy() * std_y + mu_y
    eval_m("Direct Deep MLP (1536-d)", p_mlp)
