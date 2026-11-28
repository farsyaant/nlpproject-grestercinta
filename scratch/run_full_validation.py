import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

print("=== Menjalankan Validasi Penuh Pipeline Enhanced ===")

DATA_PATH = Path("data/processed/merged_data_clean.csv")
df_raw = pd.read_csv(DATA_PATH)
date_col = "Tanggal_dt" if "Tanggal_dt" in df_raw.columns else "tanggal"
df_raw[date_col] = pd.to_datetime(df_raw[date_col])
df_raw = df_raw.sort_values(date_col).reset_index(drop=True)

if "Kurs Jual" in df_raw.columns and "Kurs Beli" in df_raw.columns:
    df_raw["rate_clean"] = (df_raw["Kurs Jual"] + df_raw["Kurs Beli"]) / 2.0
elif "rate_clean" not in df_raw.columns and "rate" in df_raw.columns:
    df_raw["rate_clean"] = df_raw["rate"]

def clean_text_fn(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

title_col = "all_title" if "all_title" in df_raw.columns else "title"
body_col  = "all_full_text" if "all_full_text" in df_raw.columns else "full_text"

df_raw["title_clean"]    = df_raw[title_col].apply(clean_text_fn)
df_raw["fulltext_clean"] = df_raw[body_col].apply(clean_text_fn)

unique_dates = df_raw[date_col].drop_duplicates().sort_values().reset_index(drop=True)
n_dates = len(unique_dates)
train_end = unique_dates.iloc[int(n_dates * 0.70) - 1]
val_end   = unique_dates.iloc[int(n_dates * 0.85) - 1]

# 1. GPR Features
GPR_TERMS = [
    "war", "military", "conflict", "geopolitic", "geopolitical", "threat", "sanction", "sanctions",
    "tension", "tensions", "crisis", "attack", "invasion", "nuclear", "terrorism", "terrorist",
    "missile", "hostility", "clash", "rebel", "insurgency", "troops", "sovereignty", "escalation",
    "retaliation", "dispute", "standoff", "blockade", "weapon", "drills", "border", "combat",
    "perang", "militer", "konflik", "geopolitik", "ancaman", "sanksi", "ketegangan", "krisis",
    "serangan", "invasi", "nuklir", "terorisme", "teroris", "rudal", "bentrokan", "pemberontak",
    "pasukan", "kedaulatan", "eskalasi", "pembalasan", "sengketa", "senjata", "perbatasan"
]

def extract_gpr(title, body):
    combined = (str(title) + " " + str(body)).lower()
    tokens = combined.split()
    total_tokens = len(tokens)
    if total_tokens == 0:
        return 0, 0.0
    c = sum(1 for tok in tokens if any(term in tok for term in GPR_TERMS))
    return c, c / total_tokens

gpr_metrics = [extract_gpr(t, b) for t, b in zip(df_raw["title_clean"], df_raw["fulltext_clean"])]
df_raw["gpr_count"]   = [m[0] for m in gpr_metrics]
df_raw["gpr_density"] = [m[1] for m in gpr_metrics]

# 2. Sentimen FinBERT
sent_cache = Path("data/processed/finbert_sentiment.csv")
if sent_cache.exists():
    sent_df = pd.read_csv(sent_cache)
    for col in ["sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_polarity"]:
        df_raw[col] = sent_df[col]
else:
    df_raw["sentiment_pos"] = 0.33
    df_raw["sentiment_neg"] = 0.33
    df_raw["sentiment_neu"] = 0.34
    df_raw["sentiment_polarity"] = 0.0

# 3. Topik SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

tr_docs = (df_raw.loc[df_raw[date_col] <= train_end, "title_clean"] + " " + df_raw.loc[df_raw[date_col] <= train_end, "fulltext_clean"]).tolist()
all_docs = (df_raw["title_clean"] + " " + df_raw["fulltext_clean"]).tolist()
vec = TfidfVectorizer(max_features=500, stop_words="english")
X_tr_tfidf = vec.fit_transform(tr_docs)
svd = TruncatedSVD(n_components=5, random_state=SEED)
svd.fit(X_tr_tfidf)
topics = np.clip(svd.transform(vec.transform(all_docs)), 0, None)
for k in range(5):
    df_raw[f"topic_prob_{k}"] = topics[:, k]

# 4. Agregasi Harian & Fitur Temporal
text_daily = df_raw.groupby(date_col).agg({
    "rate_clean": "first",
    "gpr_count": "mean",
    "gpr_density": "mean",
    "sentiment_pos": "mean",
    "sentiment_neg": "mean",
    "sentiment_neu": "mean",
    "sentiment_polarity": "mean",
    "topic_prob_0": "mean",
    "topic_prob_1": "mean",
    "topic_prob_2": "mean",
    "topic_prob_3": "mean",
    "topic_prob_4": "mean"
})

text_daily["ret"]      = text_daily["rate_clean"].pct_change()
text_daily["ret_lag1"] = text_daily["ret"].shift(1)
text_daily["ret_lag2"] = text_daily["ret"].shift(2)
text_daily["ret_lag3"] = text_daily["ret"].shift(3)
text_daily["vol_5d"]   = text_daily["ret"].rolling(5).std()
text_daily["vol_20d"]  = text_daily["ret"].rolling(20).std()
text_daily["mom_5d"]   = text_daily["rate_clean"] / text_daily["rate_clean"].shift(5) - 1.0
text_daily["mom_20d"]  = text_daily["rate_clean"] / text_daily["rate_clean"].shift(20) - 1.0
text_daily["dow"]      = text_daily.index.dayofweek

text_daily["ret_magnitude"] = text_daily["ret"].abs()
text_daily["ret_direction"] = (text_daily["ret"] > 0).astype(int)

daily_valid = text_daily.dropna(subset=["ret", "ret_lag3", "vol_20d", "mom_20d"]).copy()

tr_m = daily_valid.index <= train_end
va_m = (daily_valid.index > train_end) & (daily_valid.index <= val_end)
te_m = daily_valid.index > val_end

text_feat_cols = [
    "gpr_count", "gpr_density", "sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_polarity",
    "topic_prob_0", "topic_prob_1", "topic_prob_2", "topic_prob_3", "topic_prob_4"
]

scaler_txt = StandardScaler().fit(daily_valid.loc[tr_m, text_feat_cols])
X_txt_tr = scaler_txt.transform(daily_valid.loc[tr_m, text_feat_cols])
X_txt_all = scaler_txt.transform(daily_valid[text_feat_cols])

# Stage 1a: Magnitude
mlp_mag = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", alpha=0.1, max_iter=250, random_state=SEED)
mlp_mag.fit(X_txt_tr, daily_valid.loc[tr_m, "ret_magnitude"].values)
daily_valid["pred_text_magnitude"] = mlp_mag.predict(X_txt_all)

# Stage 1b: Direction Classifier
clf_dir = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", alpha=0.5, max_iter=250, random_state=SEED, early_stopping=True)
clf_dir.fit(X_txt_tr, daily_valid.loc[tr_m, "ret_direction"].values)
daily_valid["pred_text_prob_up"] = clf_dir.predict_proba(X_txt_all)[:, 1]
daily_valid["pred_text_signed_signal"] = (2 * daily_valid["pred_text_prob_up"] - 1.0) * daily_valid["pred_text_magnitude"]

# Stage 2: Master Regressors
temporal_cols = ["ret_lag1", "ret_lag2", "ret_lag3", "vol_5d", "vol_20d", "mom_5d", "mom_20d", "dow"]
signal_cols = ["pred_text_magnitude", "pred_text_prob_up", "pred_text_signed_signal"]
master_cols = temporal_cols + signal_cols + text_feat_cols

scaler_master = StandardScaler().fit(daily_valid.loc[tr_m, master_cols])
X_tr = scaler_master.transform(daily_valid.loc[tr_m, master_cols])
X_te = scaler_master.transform(daily_valid.loc[te_m, master_cols])

y_tr = daily_valid.loc[tr_m, "ret"].values
y_te = daily_valid.loc[te_m, "ret"].values
mu_y, std_y = y_tr.mean(), y_tr.std()
y_tr_s = (y_tr - mu_y) / std_y

def compute_metrics(y_true, y_pred, y_ref):
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    denom = np.sum((y_true - y_ref) ** 2)
    r2 = float(1 - np.sum(err ** 2) / denom) if denom > 0 else 0.0
    valid_sign = (y_true != 0) & (y_pred != 0)
    da = float(np.mean(np.sign(y_true[valid_sign]) == np.sign(y_pred[valid_sign])) * 100) if valid_sign.any() else 50.0
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "DirAcc (%)": da, "R2_oos": r2}

# Master Models
mlp1 = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", alpha=1.0, max_iter=300, random_state=SEED)
mlp1.fit(X_tr, y_tr_s)
p_mlp1 = mlp1.predict(X_te) * std_y + mu_y

mlp2 = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", alpha=2.0, max_iter=300, random_state=SEED)
mlp2.fit(X_tr, y_tr_s)
p_mlp2 = mlp2.predict(X_te) * std_y + mu_y

ridge = Ridge(alpha=10.0, random_state=SEED)
ridge.fit(X_tr, y_tr_s)
p_ridge = ridge.predict(X_te) * std_y + mu_y

p_blend = 0.5 * p_mlp1 + 0.5 * p_ridge

res = [
    {"Model": "Baseline: Tebak Return 0",         **compute_metrics(y_te, np.zeros_like(y_te), mu_y)},
    {"Model": "Baseline: Tebak Rata-rata Train",  **compute_metrics(y_te, np.full_like(y_te, mu_y), mu_y)},
    {"Model": "Master MLP Regressor (alpha=1.0)", **compute_metrics(y_te, p_mlp1, mu_y)},
    {"Model": "Master MLP Regressor (alpha=2.0)", **compute_metrics(y_te, p_mlp2, mu_y)},
    {"Model": "Master Ridge Regressor (Shrinkage)",**compute_metrics(y_te, p_ridge, mu_y)},
    {"Model": "Ensemble Blend (MLP + Ridge)",      **compute_metrics(y_te, p_blend, mu_y)}
]

res_df = pd.DataFrame(res).set_index("Model")
print("\n" + "="*80)
print("HASIL LENGKAP EVALUASI TEST SET:")
print(res_df.to_string())
print("="*80)

# Simpan laporan CSV
reports_dir = Path("percobaan task 3/reports")
reports_dir.mkdir(parents=True, exist_ok=True)
csv_out = reports_dir / "evaluasi_rmse_002xxxx.csv"
res_df.to_csv(csv_out)
print(f"Laporan berhasil disimpan di: {csv_out}")
