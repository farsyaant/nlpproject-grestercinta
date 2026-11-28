import os
import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, roc_auc_score

print("=== Eksperimen Integrasi Text Direction + Magnitude ===")

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

# 2. Fitur Geopolitik (GPR)
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

# 3. Fitur Sentimen & Polaritas
finbert_cache = 'data/processed/finbert_sentiment.csv'
if os.path.exists(finbert_cache):
    sent_df = pd.read_csv(finbert_cache)
    for col in ['sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_polarity']:
        if col in sent_df.columns:
            df[col] = sent_df[col]
else:
    df['sentiment_pos'] = 0.33
    df['sentiment_neg'] = 0.33
    df['sentiment_neu'] = 0.34
    df['sentiment_polarity'] = 0.0

# 4. Target
df['ret_magnitude'] = df['ret'].abs()
df['ret_direction'] = (df['ret'] > 0).astype(int) # 1 = Dolar Menguat / Rupiah Melemah

# Filter baris yang valid
valid_df = df.dropna(subset=['ret', 'ret_lag3', 'vol_20d', 'mom_20d']).reset_index(drop=True)
N = len(valid_df)
n_train = int(N * 0.70)
n_val = int(N * 0.15)
n_test = N - n_train - n_val

train_df = valid_df.iloc[:n_train].copy()
val_df   = valid_df.iloc[n_train:n_train+n_val].copy()
test_df  = valid_df.iloc[n_train+n_val:].copy()

text_cols = ['gpr_count', 'gpr_density', 'sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_polarity']
scaler_txt = StandardScaler().fit(train_df[text_cols])
X_txt_tr = scaler_txt.transform(train_df[text_cols])
X_txt_va = scaler_txt.transform(val_df[text_cols])
X_txt_te = scaler_txt.transform(test_df[text_cols])

# -------------------------------------------------------------
# A. Stage 1a: Text-MLP Regressor (Magnitude: |ret|)
# -------------------------------------------------------------
y_mag_tr = train_df['ret_magnitude'].values
y_mag_va = val_df['ret_magnitude'].values
y_mag_te = test_df['ret_magnitude'].values

mlp_mag = MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', alpha=0.1,
                       learning_rate_init=0.001, max_iter=250, random_state=42)
mlp_mag.fit(X_txt_tr, y_mag_tr)

train_df['pred_text_magnitude'] = mlp_mag.predict(X_txt_tr)
val_df['pred_text_magnitude']   = mlp_mag.predict(X_txt_va)
test_df['pred_text_magnitude']  = mlp_mag.predict(X_txt_te)

# -------------------------------------------------------------
# B. Stage 1b: Text Direction Classifier (Melemah vs Menguat)
# -------------------------------------------------------------
y_dir_tr = train_df['ret_direction'].values
y_dir_va = val_df['ret_direction'].values
y_dir_te = test_df['ret_direction'].values

# Latih Classifier untuk memprediksi probabilitas Rupiah Melemah (USD naik)
text_clf = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', alpha=0.5,
                         learning_rate_init=0.001, max_iter=250, random_state=42,
                         early_stopping=True, validation_fraction=0.15)
text_clf.fit(X_txt_tr, y_dir_tr)

train_df['pred_text_prob_up'] = text_clf.predict_proba(X_txt_tr)[:, 1]
val_df['pred_text_prob_up']   = text_clf.predict_proba(X_txt_va)[:, 1]
test_df['pred_text_prob_up']  = text_clf.predict_proba(X_txt_te)[:, 1]

# Signed Score: kombinasi magnitude x probabilitas arah (-1 s/d +1)
train_df['pred_text_signed_signal'] = (2 * train_df['pred_text_prob_up'] - 1.0) * train_df['pred_text_magnitude']
val_df['pred_text_signed_signal']   = (2 * val_df['pred_text_prob_up'] - 1.0) * val_df['pred_text_magnitude']
test_df['pred_text_signed_signal']  = (2 * test_df['pred_text_prob_up'] - 1.0) * test_df['pred_text_magnitude']

text_acc_te = accuracy_score(y_dir_te, (test_df['pred_text_prob_up'] >= 0.5).astype(int))
print(f"Stage 1b Text Direction Accuracy di Test Set: {text_acc_te*100:.2f}%")

# -------------------------------------------------------------
# C. Stage 2: Master Dataset & Master MLP/Ridge Modeling
# -------------------------------------------------------------
temporal_cols = ['ret_lag1', 'ret_lag2', 'ret_lag3', 'vol_5d', 'vol_20d', 'mom_5d', 'mom_20d', 'dow']
signal_cols = ['pred_text_magnitude', 'pred_text_prob_up', 'pred_text_signed_signal']
all_features = temporal_cols + signal_cols + text_cols

scaler_master = StandardScaler().fit(train_df[all_features])
X_tr = scaler_master.transform(train_df[all_features])
X_va = scaler_master.transform(val_df[all_features])
X_te = scaler_master.transform(test_df[all_features])

y_tr = train_df['ret'].values
y_va = val_df['ret'].values
y_te = test_df['ret'].values

mu_tr = np.mean(y_tr)
std_tr = np.std(y_tr)

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

print("\n--- HASIL EVALUASI MASTER STAGE 2 DENGAN FITUR TEXT DIRECTION ---")
evaluate(y_te, np.zeros_like(y_te), "Baseline 0")
evaluate(y_te, np.full_like(y_te, mu_tr), "Baseline Mean")

# 1. Master MLP dengan berbagai alpha
for a in [0.1, 0.5, 1.0, 2.0]:
    mlp = MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', alpha=a,
                       learning_rate_init=0.001, max_iter=300, random_state=42,
                       early_stopping=True, validation_fraction=0.15, n_iter_no_change=20)
    mlp.fit(X_tr, y_tr_s)
    p_mlp = mlp.predict(X_te) * std_tr + mu_tr
    evaluate(y_te, p_mlp, f"Master MLP (alpha={a})")

# 2. Ridge Regression
for a in [1.0, 10.0, 50.0, 100.0]:
    ridge = Ridge(alpha=a, random_state=42)
    ridge.fit(X_tr, y_tr_s)
    p_ridge = ridge.predict(X_te) * std_tr + mu_tr
    evaluate(y_te, p_ridge, f"Master Ridge (alpha={a})")

# 3. Ensemble Blend (MLP + Ridge)
mlp_best = MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', alpha=1.0, max_iter=300, random_state=42)
mlp_best.fit(X_tr, y_tr_s)
p_mlp = mlp_best.predict(X_te) * std_tr + mu_tr

ridge_best = Ridge(alpha=10.0, random_state=42)
ridge_best.fit(X_tr, y_tr_s)
p_ridge = ridge_best.predict(X_te) * std_tr + mu_tr

p_blend = 0.5 * p_mlp + 0.5 * p_ridge
evaluate(y_te, p_blend, "Ensemble Blend (MLP + Ridge)")
