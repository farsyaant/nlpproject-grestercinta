import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

# 1. Muat Data
df_raw = pd.read_csv('data/processed/merged_data_clean.csv')
date_col = 'Tanggal_dt' if 'Tanggal_dt' in df_raw.columns else 'tanggal'
df_raw[date_col] = pd.to_datetime(df_raw[date_col])
df_raw = df_raw.sort_values(date_col).reset_index(drop=True)

if 'Kurs Jual' in df_raw.columns and 'Kurs Beli' in df_raw.columns:
    df_raw['rate_clean'] = (df_raw['Kurs Jual'] + df_raw['Kurs Beli']) / 2.0
elif 'rate_clean' not in df_raw.columns and 'rate' in df_raw.columns:
    df_raw['rate_clean'] = df_raw['rate']

# 2. Definisikan Kata Kunci Entitas & GPR
US_TERMS = [
    r"\bus\b", r"\bu\.s\.\b", r"\bamerica\b", r"\bamerican\b", r"\bfed\b", r"\bfederal reserve\b",
    r"\bpowell\b", r"\bdollar\b", r"\bgreenback\b", r"\bwall street\b", r"\btreasury\b", r"\bwashington\b"
]
IDR_TERMS = [
    r"\bindonesia\b", r"\bindonesian\b", r"\bjakarta\b", r"\brupiah\b", r"\bbi\b", r"\bbank indonesia\b",
    r"\bjokowi\b", r"\bprabowo\b", r"\bsri mulyani\b", r"\bihsg\b", r"\bbumn\b"
]
GPR_TERMS = [
    "war", "military", "conflict", "geopolitic", "threat", "sanction", "sanctions",
    "tension", "crisis", "attack", "invasion", "nuclear", "terrorism", "missile",
    "perang", "militer", "konflik", "geopolitik", "ancaman", "sanksi", "ketegangan", "krisis"
]

def extract_entity_and_gpr(title, body):
    text = (str(title) + " " + str(body)).lower()
    
    # Flag Entitas
    has_us = int(any(re.search(pat, text) for pat in US_TERMS))
    has_idr = int(any(re.search(pat, text) for pat in IDR_TERMS))
    
    # GPR
    toks = text.split()
    total = len(toks) if toks else 1
    gpr_c = sum(1 for tok in toks if any(g in tok for g in GPR_TERMS))
    gpr_density = gpr_c / total
    
    return has_us, has_idr, gpr_c, gpr_density

features_list = [extract_entity_and_gpr(t, b) for t, b in zip(df_raw['all_title'], df_raw['all_full_text'])]
df_raw['flag_us'] = [f[0] for f in features_list]
df_raw['flag_idr'] = [f[1] for f in features_list]
df_raw['gpr_count'] = [f[2] for f in features_list]
df_raw['gpr_density'] = [f[3] for f in features_list]

# Muat / Buat Sentimen FinBERT jika ada, atau gunakan leksikon Loughran-McDonald
sent_cache = 'data/processed/finbert_sentiment.csv'
if os.path.exists(sent_cache):
    s_df = pd.read_csv(sent_cache)
    df_raw['sentiment_pos'] = s_df['sentiment_pos']
    df_raw['sentiment_neg'] = s_df['sentiment_neg']
    df_raw['sentiment_polarity'] = s_df['sentiment_polarity']
else:
    # Menggunakan leksikon Loughran-McDonald financial sentiment sebagai proxy cepat jika finbert belum di-cache
    lm_path = 'data/raw/Loughran-McDonald_MasterDictionary_1993-2025.csv'
    if os.path.exists(lm_path):
        lm_dict = pd.read_csv(lm_path)
        pos_words = set(lm_dict[lm_dict['Positive'] > 0]['Word'].str.lower().dropna())
        neg_words = set(lm_dict[lm_dict['Negative'] > 0]['Word'].str.lower().dropna())
        
        pos_scores, neg_scores = [], []
        for text in df_raw['all_title']:
            toks = str(text).lower().split()
            p_c = sum(1 for t in toks if t in pos_words)
            n_c = sum(1 for t in toks if t in neg_words)
            pos_scores.append(p_c)
            neg_scores.append(n_c)
        df_raw['sentiment_pos'] = np.array(pos_scores) / 10.0
        df_raw['sentiment_neg'] = np.array(neg_scores) / 10.0
        df_raw['sentiment_polarity'] = df_raw['sentiment_pos'] - df_raw['sentiment_neg']
    else:
        df_raw['sentiment_pos'] = 0.5
        df_raw['sentiment_neg'] = 0.5
        df_raw['sentiment_polarity'] = 0.0

# Pastikan polaritas strictly biner/bipolar (-1 atau +1) sesuai request pengguna
# Jika polarity == 0, fallback ke 0
df_raw['polarity_sign'] = np.sign(df_raw['sentiment_polarity'])

# Sinyal Arah Tertarget (Target-Based Directional Impact)
# US Positif -> Kurs Naik (+1). IDR Positif -> Kurs Turun (-1).
# GPR Tinggi + Negatif -> Melemah (+1 untuk USD/IDR).
df_raw['entity_directional_signal'] = (
    df_raw['flag_us'] * df_raw['polarity_sign'] - 
    df_raw['flag_idr'] * df_raw['polarity_sign']
)
df_raw['gpr_weakening_signal'] = df_raw['gpr_density'] * np.where(df_raw['polarity_sign'] < 0, 1.0, -0.5)

print("Statistik Fitur Teks Tertarget:")
print(df_raw[['flag_us', 'flag_idr', 'polarity_sign', 'entity_directional_signal', 'gpr_weakening_signal']].describe().round(4))

# 3. Agregasi ke Level Harian
daily = df_raw.groupby(date_col).agg({
    'rate_clean': 'first',
    'flag_us': 'mean',
    'flag_idr': 'mean',
    'polarity_sign': 'mean',
    'entity_directional_signal': 'mean',
    'gpr_weakening_signal': 'mean',
    'gpr_density': 'mean'
})

# Target: log_return
daily['log_ret'] = np.log(daily['rate_clean'] / daily['rate_clean'].shift(1))
# Target Arah: 1 = USD Naik / Rupiah Melemah (log_ret > 0), 0 = IDR Naik / Rupiah Menguat (log_ret <= 0)
daily['target_direction'] = (daily['log_ret'] > 0).astype(int)

# 4. Fitur Deret Waktu Ringkas (Basic Time-Series strictly lagged t-1)
daily['ret_lag1'] = daily['log_ret'].shift(1)
daily['ret_lag2'] = daily['log_ret'].shift(2)
daily['vol_5d']   = daily['log_ret'].shift(1).rolling(5).std()

# Reversal & Momentum yang diminta:
# - consecutive_up_days: berapa hari kurs sudah naik berturut-turut (negatif jika turun berturut-turut)
consecs = []
cur = 0
for r in daily['log_ret']:
    if pd.isna(r):
        consecs.append(0)
    elif r > 0:
        cur = cur + 1 if cur > 0 else 1
        consecs.append(cur)
    elif r < 0:
        cur = cur - 1 if cur < 0 else -1
        consecs.append(cur)
    else:
        cur = 0
        consecs.append(0)
daily['consec_up_days'] = pd.Series(consecs, index=daily.index).shift(1)

# - dist_ma20: Selisih jarak terhadap Moving Average 20 hari (rate / MA20 - 1) dihitung hingga t-1
ma20 = daily['rate_clean'].shift(1).rolling(20).mean()
daily['dist_ma20'] = daily['rate_clean'].shift(1) / ma20 - 1.0

# Filter baris lengkap
valid = daily.dropna(subset=['log_ret', 'ret_lag2', 'vol_5d', 'dist_ma20', 'consec_up_days']).copy()
print(f"\nTotal baris harian valid: {len(valid)}")

# Pembagian Kronologis (70% Train, 15% Val, 15% Test)
n = len(valid)
n_tr = int(n * 0.70)
n_va = int(n * 0.15)
tr_mask = np.zeros(n, dtype=bool)
va_mask = np.zeros(n, dtype=bool)
te_mask = np.zeros(n, dtype=bool)
tr_mask[:n_tr] = True
va_mask[n_tr:n_tr+n_va] = True
te_mask[n_tr+n_va:] = True

print(f"Data Train: {tr_mask.sum()}, Val: {va_mask.sum()}, Test: {te_mask.sum()}")

# 5. Pemodelan Tahap 1: Klasifikasi Arah (USD vs IDR) dari Fitur Teks dengan OOF
text_features = ['flag_us', 'flag_idr', 'polarity_sign', 'entity_directional_signal', 'gpr_weakening_signal', 'gpr_density']
scaler_txt = StandardScaler().fit(valid.loc[tr_mask, text_features])
X_txt_tr = scaler_txt.transform(valid.loc[tr_mask, text_features])
X_txt_va = scaler_txt.transform(valid.loc[va_mask, text_features])
X_txt_te = scaler_txt.transform(valid.loc[te_mask, text_features])

y_dir_tr = valid.loc[tr_mask, 'target_direction'].values
y_dir_te = valid.loc[te_mask, 'target_direction'].values

# OOF 5-Fold pada Train set
kf = KFold(n_splits=5, shuffle=False)
oof_dir_prob = np.zeros(len(X_txt_tr))
for fold_tr, fold_val in kf.split(X_txt_tr):
    clf = LogisticRegression(C=0.5, random_state=42)
    clf.fit(X_txt_tr[fold_tr], y_dir_tr[fold_tr])
    oof_dir_prob[fold_val] = clf.predict_proba(X_txt_tr[fold_val])[:, 1]

# Fit model penuh untuk Val & Test
clf_full = LogisticRegression(C=0.5, random_state=42).fit(X_txt_tr, y_dir_tr)
pred_prob_va = clf_full.predict_proba(X_txt_va)[:, 1]
pred_prob_te = clf_full.predict_proba(X_txt_te)[:, 1]

valid.loc[tr_mask, 'pred_text_prob_up'] = oof_dir_prob
valid.loc[va_mask, 'pred_text_prob_up'] = pred_prob_va
valid.loc[te_mask, 'pred_text_prob_up'] = pred_prob_te

acc_stage1 = accuracy_score(y_dir_te, (pred_prob_te >= 0.5).astype(int)) * 100
print(f"\nAkurasi Stage 1 (Klasifikasi Arah dari Teks pada Test Set): {acc_stage1:.2f}%")

# 6. Pemodelan Tahap 2: Prediksi log_return (Gabungan Teks + Fitur Time-Series Basic)
ts_features = ['ret_lag1', 'ret_lag2', 'vol_5d', 'consec_up_days', 'dist_ma20']
stage2_features = ts_features + ['pred_text_prob_up', 'entity_directional_signal', 'gpr_weakening_signal']

sc_s2 = StandardScaler().fit(valid.loc[tr_mask, stage2_features])
X_s2_tr = sc_s2.transform(valid.loc[tr_mask, stage2_features])
X_s2_te = sc_s2.transform(valid.loc[te_mask, stage2_features])

y_log_tr = valid.loc[tr_mask, 'log_ret'].values
y_log_te = valid.loc[te_mask, 'log_ret'].values

mu_log, std_log = y_log_tr.mean(), y_log_tr.std()
y_log_tr_s = (y_log_tr - mu_log) / std_log

# Baseline Mean
pred_base_te = np.full_like(y_log_te, mu_log)

# Model 1: Basic MLP Regressor
mlp_basic = MLPRegressor(hidden_layer_sizes=(16, 8), alpha=2.0, max_iter=200, random_state=42)
mlp_basic.fit(X_s2_tr, y_log_tr_s)
pred_mlp_te = mlp_basic.predict(X_s2_te) * std_log + mu_log

# Model 2: Ridge Regressor
ridge_s2 = Ridge(alpha=50.0, random_state=42)
ridge_s2.fit(X_s2_tr, y_log_tr_s)
pred_ridge_te = ridge_s2.predict(X_s2_te) * std_log + mu_log

# Evaluasi
def eval_metrics(y_true, y_pred, name):
    err = y_true - y_pred
    mae = np.mean(np.abs(err))
    mse = np.mean(err**2)
    rmse = np.sqrt(mse)
    da = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
    print(f"{name:30s}: RMSE={rmse:.6f}, MAE={mae:.6f}, MSE={mse:.8f}, DirAcc={da:.2f}%")

print("\n--- HASIL EVALUASI TAHAP 2 (PREDIKSI LOG_RETURN) ---")
eval_metrics(y_log_te, pred_base_te, "Baseline Mean")
eval_metrics(y_log_te, pred_mlp_te, "Basic MLP Regressor")
eval_metrics(y_log_te, pred_ridge_te, "Ridge Regressor")
