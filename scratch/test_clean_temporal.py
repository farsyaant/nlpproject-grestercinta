import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('data/processed/merged_data_clean.csv')
if 'Kurs Jual' in df.columns and 'Kurs Beli' in df.columns:
    df['rate'] = (df['Kurs Jual'] + df['Kurs Beli']) / 2.0
elif 'rate_clean' in df.columns:
    df['rate'] = df['rate_clean']

df['Tanggal_dt'] = pd.to_datetime(df['Tanggal_dt'])
df = df.sort_values('Tanggal_dt').reset_index(drop=True)

# Return hari t
df['ret'] = df['rate'].pct_change()

# FITUR TEMPORAL YANG BENAR (Semua diketahui di t-1)
df['ret_lag1'] = df['ret'].shift(1)
df['ret_lag2'] = df['ret'].shift(2)
df['ret_lag3'] = df['ret'].shift(3)
df['vol_5d_clean']  = df['ret'].shift(1).rolling(5).std()
df['vol_20d_clean'] = df['ret'].shift(1).rolling(20).std()
df['mom_5d_clean']  = df['rate'].shift(1) / df['rate'].shift(6) - 1.0
df['mom_20d_clean'] = df['rate'].shift(1) / df['rate'].shift(21) - 1.0
df['dow'] = df['Tanggal_dt'].dt.dayofweek

# Bandingkan dengan versi leaky sebelumnya:
df['vol_5d_leaky'] = df['ret'].rolling(5).std()
df['mom_5d_leaky'] = df['rate'] / df['rate'].shift(5) - 1.0

valid = df.dropna(subset=['ret', 'vol_20d_clean', 'mom_20d_clean']).reset_index(drop=True)
N = len(valid)
n_train = int(N * 0.70)
n_val = int(N * 0.15)

tr = valid.iloc[:n_train]
te = valid.iloc[n_train+n_val:]

y_tr = tr['ret'].values
y_te = te['ret'].values
mu_tr = y_tr.mean()
std_tr = y_tr.std()

print("--- EVALUASI FITUR TEMPORAL TANPA LEAKAGE (Semua Lagged t-1) ---")
cols_clean = ['ret_lag1', 'ret_lag2', 'ret_lag3', 'vol_5d_clean', 'vol_20d_clean', 'mom_5d_clean', 'mom_20d_clean', 'dow']
scaler = StandardScaler().fit(tr[cols_clean])
X_tr_c = scaler.transform(tr[cols_clean])
X_te_c = scaler.transform(te[cols_clean])

ridge = Ridge(alpha=100.0, random_state=42)
ridge.fit(X_tr_c, (y_tr - mu_tr) / std_tr)
p_ridge = ridge.predict(X_te_c) * std_tr + mu_tr

rmse = np.sqrt(mean_squared_error(y_te, p_ridge))
mae = mean_absolute_error(y_te, p_ridge)
sign_valid = (y_te != 0) & (p_ridge != 0)
diracc = np.mean(np.sign(y_te[sign_valid]) == np.sign(p_ridge[sign_valid])) * 100
r2 = 1.0 - np.sum((y_te - p_ridge)**2) / np.sum((y_te - mu_tr)**2)

print(f"Clean Temporal Ridge: RMSE={rmse:.6f}, MAE={mae:.6f}, DirAcc={diracc:.2f}%, R2_oos={r2:.5f}")

# Bandingkan Baseline:
print(f"Baseline Tebak 0    : RMSE={np.sqrt(np.mean(y_te**2)):.6f}, MAE={np.mean(np.abs(y_te)):.6f}")
print(f"Baseline Mean Train : RMSE={np.sqrt(np.mean((y_te - mu_tr)**2)):.6f}, MAE={np.mean(np.abs(y_te - mu_tr)):.6f}")
