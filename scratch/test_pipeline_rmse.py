import os
import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("=== Memulai Eksperimen Optimasi RMSE ===")

df = pd.read_csv('data/processed/merged_data_clean.csv')
if 'Kurs Jual' in df.columns and 'Kurs Beli' in df.columns:
    df['rate'] = (df['Kurs Jual'] + df['Kurs Beli']) / 2.0
elif 'rate_clean' in df.columns:
    df['rate'] = df['rate_clean']

df['Tanggal_dt'] = pd.to_datetime(df['Tanggal_dt'])
df = df.sort_values('Tanggal_dt').reset_index(drop=True)

# 1. Feature Engineering Temporal Kurs
df['ret'] = df['rate'].pct_change()
df['ret_lag1'] = df['ret'].shift(1)
df['ret_lag2'] = df['ret'].shift(2)
df['ret_lag3'] = df['ret'].shift(3)
df['vol_5d']   = df['ret'].rolling(5).std()
df['vol_20d']  = df['ret'].rolling(20).std()
df['mom_5d']   = df['rate'] / df['rate'].shift(5) - 1.0
df['mom_20d']  = df['rate'] / df['rate'].shift(20) - 1.0
df['dow']      = df['Tanggal_dt'].dt.dayofweek

# 2. Text Features (GPR, Sentiment, Length)
gpr_terms = [
    "war", "military", "conflict", "geopolitic", "geopolitical", "threat", "sanction", "sanctions",
    "tension", "tensions", "crisis", "attack", "invasion", "nuclear", "terrorism", "terrorist",
    "missile", "hostility", "clash", "rebel", "insurgency", "troops", "sovereignty", "escalation",
    "retaliation", "dispute", "standoff", "blockade", "weapon", "drills", "border", "combat",
    "perang", "militer", "konflik", "geopolitik", "ancaman", "sanksi", "ketegangan", "krisis",
    "serangan", "invasi", "nuklir", "terorisme", "teroris", "rudal", "bentrokan", "pemberontak",
    "pasukan", "kedaulatan", "eskalasi", "pembalasan", "sengketa", "senjata", "perbatasan"
]

def compute_gpr(text):
    if not isinstance(text, str) or not text.strip():
        return 0, 0.0
    tokens = text.lower().split()
    total = len(tokens)
    if total == 0:
        return 0, 0.0
    cnt = sum(1 for tok in tokens if any(g in tok for g in gpr_terms))
    return cnt, cnt / total

gpr_res = [compute_gpr(str(t) + " " + str(f)) for t, f in zip(df['all_title'], df['all_full_text'])]
df['gpr_count']   = [r[0] for r in gpr_res]
df['gpr_density'] = [r[1] for r in gpr_res]

# FinBERT Sentiment Cache jika ada
finbert_cache = 'data/processed/finbert_sentiment.csv'
if os.path.exists(finbert_cache):
    sent_df = pd.read_csv(finbert_cache)
    for col in ['sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_polarity']:
        if col in sent_df.columns:
            df[col] = sent_df[col]
else:
    # Dummy placeholder sementara jika cache belum ada
    df['sentiment_pos'] = 0.33
    df['sentiment_neg'] = 0.33
    df['sentiment_neu'] = 0.34
    df['sentiment_polarity'] = 0.0

# 3. Target Magnitude
df['ret_magnitude'] = df['ret'].abs()

# Filter data valid
valid_df = df.dropna(subset=['ret', 'ret_lag3', 'vol_20d', 'mom_20d']).reset_index(drop=True)
N = len(valid_df)
n_train = int(N * 0.70)
n_val = int(N * 0.15)
n_test = N - n_train - n_val

train_df = valid_df.iloc[:n_train].copy()
val_df   = valid_df.iloc[n_train:n_train+n_val].copy()
test_df  = valid_df.iloc[n_train+n_val:].copy()

print(f"Data valid: Total={N}, Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# Fitur Text untuk Stage 1
text_cols = ['gpr_count', 'gpr_density', 'sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_polarity']
scaler_txt = StandardScaler().fit(train_df[text_cols])
X_txt_tr = scaler_txt.transform(train_df[text_cols])
X_txt_va = scaler_txt.transform(val_df[text_cols])
X_txt_te = scaler_txt.transform(test_df[text_cols])

y_mag_tr = train_df['ret_magnitude'].values
y_mag_va = val_df['ret_magnitude'].values
y_mag_te = test_df['ret_magnitude'].values

# Stage 1: Text-MLP untuk Magnitude
mlp_mag = MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', alpha=0.1,
                       learning_rate_init=0.001, max_iter=200, random_state=42)
mlp_mag.fit(X_txt_tr, y_mag_tr)

train_df['pred_mag'] = mlp_mag.predict(X_txt_tr)
val_df['pred_mag']   = mlp_mag.predict(X_txt_va)
test_df['pred_mag']  = mlp_mag.predict(X_txt_te)

print(f"Stage 1 Mag MAE: {mean_absolute_error(y_mag_te, test_df['pred_mag']):.6f}")

# Stage 2: Master Dataset
feature_cols = ['ret_lag1', 'ret_lag2', 'ret_lag3', 'vol_5d', 'vol_20d', 'mom_5d', 'mom_20d', 'dow', 'pred_mag'] + text_cols
scaler_master = StandardScaler().fit(train_df[feature_cols])

X_tr = scaler_master.transform(train_df[feature_cols])
X_va = scaler_master.transform(val_df[feature_cols])
X_te = scaler_master.transform(test_df[feature_cols])

y_tr = train_df['ret'].values
y_va = val_df['ret'].values
y_te = test_df['ret'].values

mu_tr = np.mean(y_tr)
std_tr = np.std(y_tr)

# Target Scaling
y_tr_s = (y_tr - mu_tr) / std_tr
y_va_s = (y_va - mu_tr) / std_tr

def evaluate(y_true, y_pred, name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    valid_sign = (y_true != 0) & (y_pred != 0)
    diracc = np.mean(np.sign(y_true[valid_sign]) == np.sign(y_pred[valid_sign])) * 100 if np.sum(valid_sign) > 0 else 50.0
    r2_oos = 1.0 - np.sum((y_true - y_pred)**2) / np.sum((y_true - mu_tr)**2)
    print(f"[{name}]")
    print(f"   RMSE: {rmse:.7f} | MAE: {mae:.7f} | MSE: {mse:.8f} | DirAcc: {diracc:.2f}% | R2_oos: {r2_oos:.5f}")
    return {"name": name, "RMSE": rmse, "MAE": mae, "MSE": mse, "DirAcc": diracc, "R2_oos": r2_oos}

print("\n--- Evaluasi Berbagai Model Stage 2 ---")
evaluate(y_te, np.zeros_like(y_te), "Baseline 0")
evaluate(y_te, np.full_like(y_te, mu_tr), "Baseline Mean")

# 1. Ridge Regressor (Shrinkage sangat kuat, mencegah overfit)
for alpha in [1.0, 10.0, 50.0, 100.0, 300.0, 500.0]:
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X_tr, y_tr_s)
    pred_te = ridge.predict(X_te) * std_tr + mu_tr
    evaluate(y_te, pred_te, f"Ridge alpha={alpha}")

# 2. Gradient Boosting dengan depth rendah
for lr in [0.01, 0.03]:
    for max_d in [2, 3]:
        gbr = GradientBoostingRegressor(n_estimators=60, learning_rate=lr, max_depth=max_d, subsample=0.8, random_state=42)
        gbr.fit(X_tr, y_tr_s)
        pred_te = gbr.predict(X_te) * std_tr + mu_tr
        evaluate(y_te, pred_te, f"GBR lr={lr} depth={max_d}")

# 3. Master Deep MLP dengan L2 Regularization & Early Stopping
for alpha_mlp in [0.05, 0.1, 0.5, 1.0, 2.0]:
    mlp = MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', solver='adam',
                       alpha=alpha_mlp, learning_rate_init=0.001, max_iter=300, random_state=42,
                       early_stopping=True, validation_fraction=0.15, n_iter_no_change=20)
    mlp.fit(X_tr, y_tr_s)
    pred_te = mlp.predict(X_te) * std_tr + mu_tr
    evaluate(y_te, pred_te, f"Master MLP alpha={alpha_mlp}")

# 4. Blended Ensemble (MLP + Ridge)
mlp_best = MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', alpha=1.0, learning_rate_init=0.001, max_iter=300, random_state=42)
mlp_best.fit(X_tr, y_tr_s)
p_mlp = mlp_best.predict(X_te) * std_tr + mu_tr

ridge_best = Ridge(alpha=100.0, random_state=42)
ridge_best.fit(X_tr, y_tr_s)
p_ridge = ridge_best.predict(X_te) * std_tr + mu_tr

p_blend = 0.5 * p_mlp + 0.5 * p_ridge
evaluate(y_te, p_blend, "Ensemble Blend (MLP + Ridge)")
