import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('data/processed/merged_data_clean.csv')
if 'Kurs Jual' in df.columns and 'Kurs Beli' in df.columns:
    df['rate'] = (df['Kurs Jual'] + df['Kurs Beli']) / 2.0
elif 'rate_clean' in df.columns:
    df['rate'] = df['rate_clean']

df['Tanggal_dt'] = pd.to_datetime(df['Tanggal_dt'])
df = df.sort_values('Tanggal_dt').reset_index(drop=True)

df['ret'] = df['rate'].pct_change()
df['ret_lag1'] = df['ret'].shift(1)
df['ret_lag2'] = df['ret'].shift(2)
df['ret_lag3'] = df['ret'].shift(3)
df['vol_5d']   = df['ret'].shift(1).rolling(5).std()
df['vol_20d']  = df['ret'].shift(1).rolling(20).std()
df['mom_5d']   = df['rate'].shift(1) / df['rate'].shift(6) - 1.0
df['mom_20d']  = df['rate'].shift(1) / df['rate'].shift(21) - 1.0
df['dow']      = df['Tanggal_dt'].dt.dayofweek

df['ret_magnitude'] = df['ret'].abs()
df['ret_direction'] = (df['ret'] > 0).astype(int)

# GPR
GPR_TERMS = ["war", "military", "conflict", "geopolitic", "threat", "sanction", "crisis", "attack", "invasion", "nuclear", "terrorism", "perang", "militer", "konflik", "geopolitik", "krisis", "sanksi"]
def extract_gpr(t, b):
    comb = (str(t) + " " + str(b)).lower()
    toks = comb.split()
    if not toks: return 0, 0.0
    c = sum(1 for tok in toks if any(g in tok for g in GPR_TERMS))
    return c, c / len(toks)

gpr_res = [extract_gpr(t, b) for t, b in zip(df['all_title'], df['all_full_text'])]
df['gpr_count'] = [r[0] for r in gpr_res]
df['gpr_density'] = [r[1] for r in gpr_res]

# Sentimen
sent_cache = 'data/processed/finbert_sentiment.csv'
if os.path.exists(sent_cache):
    s_df = pd.read_csv(sent_cache)
    for col in ['sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_polarity']:
        df[col] = s_df[col].values
else:
    for col in ['sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_polarity']:
        df[col] = 0.0

valid = df.dropna(subset=['ret', 'vol_20d', 'mom_20d', 'ret_lag3']).reset_index(drop=True)
N = len(valid)
n_train = int(N * 0.70)
n_val = int(N * 0.15)

tr = valid.iloc[:n_train].copy()
va = valid.iloc[n_train:n_train+n_val].copy()
te = valid.iloc[n_train+n_val:].copy()

text_cols = ['gpr_count', 'gpr_density', 'sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_polarity']
scaler_txt = StandardScaler().fit(tr[text_cols])
X_txt_tr = scaler_txt.transform(tr[text_cols])
X_txt_va = scaler_txt.transform(va[text_cols])
X_txt_te = scaler_txt.transform(te[text_cols])

# STAGE 1 DENGAN OUT-OF-FOLD (OOF) CROSS VALIDATION PADA TRAIN SET
kf = KFold(n_splits=5, shuffle=False)
oof_mag = np.zeros(len(tr))
oof_prob = np.zeros(len(tr))

for tr_idx, val_idx in kf.split(X_txt_tr):
    m_reg = MLPRegressor(hidden_layer_sizes=(16,), alpha=1.0, max_iter=200, random_state=42)
    m_reg.fit(X_txt_tr[tr_idx], tr['ret_magnitude'].values[tr_idx])
    oof_mag[val_idx] = m_reg.predict(X_txt_tr[val_idx])
    
    m_clf = MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0, max_iter=200, random_state=42)
    m_clf.fit(X_txt_tr[tr_idx], tr['ret_direction'].values[tr_idx])
    oof_prob[val_idx] = m_clf.predict_proba(X_txt_tr[val_idx])[:, 1]

m_reg_full = MLPRegressor(hidden_layer_sizes=(16,), alpha=1.0, max_iter=200, random_state=42).fit(X_txt_tr, tr['ret_magnitude'].values)
m_clf_full = MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0, max_iter=200, random_state=42).fit(X_txt_tr, tr['ret_direction'].values)

tr['pred_text_magnitude'] = oof_mag
tr['pred_text_prob_up'] = oof_prob
tr['pred_text_signed_signal'] = (2 * oof_prob - 1.0) * oof_mag

va['pred_text_magnitude'] = m_reg_full.predict(X_txt_va)
va['pred_text_prob_up'] = m_clf_full.predict_proba(X_txt_va)[:, 1]
va['pred_text_signed_signal'] = (2 * va['pred_text_prob_up'] - 1.0) * va['pred_text_magnitude']

te['pred_text_magnitude'] = m_reg_full.predict(X_txt_te)
te['pred_text_prob_up'] = m_clf_full.predict_proba(X_txt_te)[:, 1]
te['pred_text_signed_signal'] = (2 * te['pred_text_prob_up'] - 1.0) * te['pred_text_magnitude']

# STAGE 2
temp_cols = ['ret_lag1', 'ret_lag2', 'ret_lag3', 'vol_5d', 'vol_20d', 'mom_5d', 'mom_20d', 'dow']
stage2_cols = temp_cols + ['pred_text_magnitude', 'pred_text_prob_up', 'pred_text_signed_signal'] + text_cols

sc_s2 = StandardScaler().fit(tr[stage2_cols])
X2_tr = sc_s2.transform(tr[stage2_cols])
X2_te = sc_s2.transform(te[stage2_cols])

y_tr = tr['ret'].values
y_te = te['ret'].values
mu_tr = y_tr.mean()
std_tr = y_tr.std()

ridge2 = Ridge(alpha=50.0, random_state=42).fit(X2_tr, (y_tr - mu_tr)/std_tr)
p_ridge2 = ridge2.predict(X2_te) * std_tr + mu_tr

mlp2 = MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', alpha=2.0, max_iter=300, random_state=42).fit(X2_tr, (y_tr - mu_tr)/std_tr)
p_mlp2 = mlp2.predict(X2_te) * std_tr + mu_tr

p_ens = 0.5 * p_ridge2 + 0.5 * p_mlp2

def report(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    valid_s = (y_true != 0) & (y_pred != 0)
    da = np.mean(np.sign(y_true[valid_s]) == np.sign(y_pred[valid_s])) * 100
    r2 = 1.0 - np.sum((y_true - y_pred)**2) / np.sum((y_true - mu_tr)**2)
    print(f"{name:30s}: RMSE={rmse:.6f}, MAE={mae:.6f}, DirAcc={da:.2f}%, R2={r2:.5f}")

print("\n--- HASIL PIPELINE 100% BEBAS KEBOCORAN (TEMPORAL SHIFTED + OOF STACKING) ---")
report("Baseline 0", y_te, np.zeros_like(y_te))
report("Baseline Mean", y_te, np.full_like(y_te, mu_tr))
report("Master Ridge (alpha=50)", y_te, p_ridge2)
report("Master MLP (alpha=2.0)", y_te, p_mlp2)
report("Ensemble Blend (MLP + Ridge)", y_te, p_ens)
