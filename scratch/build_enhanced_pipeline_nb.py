import json
import os

cells = []

def add_md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

def add_code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

# =====================================================================
# Cell 0: Markdown - Overview
# =====================================================================
add_md("""# Pipeline Eksperimen Task 3: Dual-Stage Multimodal USD/IDR Forecasting
### Integrasi Text Direction Classifier, Text Magnitude Regressor, dan Temporal Kurs

Notebook ini memuat arsitektur pipeline terpadu hasil pembaruan untuk mencapai **Test RMSE di rentang target 0.002XXXX** dan **Directional Accuracy > 70%**:

```
[News Dataset] ──┬──> [Title]     ──> BERT umum (768-d) ────────┐
                 └──> [Full-text] ──> Longformer (768-d) ───────┼─> Concat (1536-d) ──> Rolling 5d PCA
                 
                 ├──> [BERTopic]   ──> Top-5 Topic Probs ───────┤
                 ├──> [GPR Lexicon]──> Word Count & Density ────┼─> Fitur Teks Multimodal
                 └──> [FinBERT]    ──> Pos, Neg, Neu, Polarity ─┘
                                                                        │
                 ┌──────────────────────────────────────────────────────┴──────────────────────────────────────────────────────┐
                 ▼                                                                                                             ▼
   [Stage 1a: Text-MLP Regressor]                                                                              [Stage 1b: Text Direction Classifier]
   Target: Besarnya Gerakan |ret_t| (Skala Volatilitas)                                                        Target: Arah Gerakan ret_t > 0 (Melemah vs Menguat)
                 │                                                                                                             │
                 ▼                                                                                                             ▼
       [pred_text_magnitude]                                                                                     [pred_text_prob_up & pred_text_signed_signal]
                 │                                                                                                             │
                 └──────────────────────────────────────────────┬──────────────────────────────────────────────────────────────┘
                                                                │
[USD/IDR Kurs] ──> Fitur Temporal (Lag 1-3, Volatility, Momentum, DOW) ─┼──> Horizontal Master Concat
                                                                        │
                                                                        ▼
                                                   [Stage 2: Master Regressor]
                                                   - Master MLP Regressor (L2 Reg / alpha=1.0)
                                                   - Master Ridge Regressor (Shrinkage)
                                                   - Blended Ensemble (MLP + Ridge)
                                                                        │
                                                                        ▼
                                         [Evaluasi: RMSE (0.002XXXX), MAE, MSE, DirAcc, R2 OOS]
```
""")

# =====================================================================
# Cell 1 & 2: Install Dependencies
# =====================================================================
add_md("""## 0. Instalasi Library (Jika Diperlukan)
Jalankan sel ini apabila dependensi belum terinstal di environment lokal.""")

add_code("""# Instalasi library pendukung
!pip install -q -U sentence-transformers bertopic umap-learn hdbscan transformers torch scikit-learn
""")

# =====================================================================
# Cell 3 & 4: Setup Environment & Parameter
# =====================================================================
add_md("""## 1. Konfigurasi Lingkungan, Device, dan Parameter Eksperimen
Menetapkan seed deterministik, device (CUDA/CPU), serta model pretrained yang digunakan.""")

add_code("""import os
import gc
import re
import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

warnings.filterwarnings("ignore")

# Parameter Eksperimen
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TITLE_BERT_MODEL = "bert-base-uncased"
FULLTEXT_MODEL   = "allenai/longformer-base-4096"
FINBERT_MODEL    = "ProsusAI/finbert"

def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()
print(f"Device aktif      : {DEVICE}")
print(f"Pretrained Title  : {TITLE_BERT_MODEL}")
print(f"Pretrained Fulltext: {FULLTEXT_MODEL}")
print(f"Pretrained FinBERT: {FINBERT_MODEL}")
""")

# =====================================================================
# Cell 5 & 6: Load Dataset & Data Cleaning
# =====================================================================
add_md("""## 2. Pemuatan Dataset & Pemisahan Eksplisit (Title vs Full-Text)
Membaca dataset gabungan, menghitung kurs tengah $Kurs = \\frac{Jual + Beli}{2}$, membersihkan teks judul dan isi berita, serta menetapkan boundary pemisahan kronologis 70% Train, 15% Validation, dan 15% Test.""")

add_code("""# Jalur dataset utama
DATA_CANDIDATES = [
    Path("data/processed/merged_data_clean.csv"),
    Path("../data/processed/merged_data_clean.csv"),
    Path("merged_data_clean.csv")
]

DATA_PATH = None
for p in DATA_CANDIDATES:
    if p.exists():
        DATA_PATH = p
        break

if DATA_PATH is None:
    raise FileNotFoundError("File merged_data_clean.csv tidak ditemukan di direktori proyek!")

df_raw = pd.read_csv(DATA_PATH)
print(f"Dataset berhasil dimuat: {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom.")

# 1. Pastikan Tanggal terformat datetime dan terurut kronologis
date_col = "Tanggal_dt" if "Tanggal_dt" in df_raw.columns else ("tanggal" if "tanggal" in df_raw.columns else "Tanggal")
df_raw[date_col] = pd.to_datetime(df_raw[date_col])
df_raw = df_raw.sort_values(date_col).reset_index(drop=True)

# 2. Hitung Kurs Tengah Bank Indonesia
if "Kurs Jual" in df_raw.columns and "Kurs Beli" in df_raw.columns:
    df_raw["rate_clean"] = (df_raw["Kurs Jual"] + df_raw["Kurs Beli"]) / 2.0
elif "rate_clean" not in df_raw.columns and "rate" in df_raw.columns:
    df_raw["rate_clean"] = df_raw["rate"]

# 3. Pembersihan Teks Title dan Full-Text
def clean_text_fn(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

title_col = "all_title" if "all_title" in df_raw.columns else "title"
body_col  = "all_full_text" if "all_full_text" in df_raw.columns else ("all_fulltext" if "all_fulltext" in df_raw.columns else "full_text")

df_raw["title_clean"]    = df_raw[title_col].apply(clean_text_fn)
df_raw["fulltext_clean"] = df_raw[body_col].apply(clean_text_fn)

# 4. Pembagian Kronologis (70% Train, 15% Val, 15% Test)
unique_dates = df_raw[date_col].drop_duplicates().sort_values().reset_index(drop=True)
n_dates = len(unique_dates)
train_end_idx = int(n_dates * 0.70) - 1
val_end_idx   = int(n_dates * 0.85) - 1

train_end = unique_dates.iloc[train_end_idx]
val_end   = unique_dates.iloc[val_end_idx]

print(f"Total hari bursa  : {n_dates} hari")
print(f"Periode Train     : {unique_dates.iloc[0].date()} s/d {train_end.date()} ({train_end_idx + 1} hari)")
print(f"Periode Validation: {unique_dates.iloc[train_end_idx + 1].date()} s/d {val_end.date()} ({val_end_idx - train_end_idx} hari)")
print(f"Periode Test      : {unique_dates.iloc[val_end_idx + 1].date()} s/d {unique_dates.iloc[-1].date()} ({n_dates - val_end_idx - 1} hari)")
""")

# =====================================================================
# Cell 7 & 8: BERTopic Extraction
# =====================================================================
add_md("""## 3. Ekstraksi Fitur Topik (BERTopic)
Mengekstraksi representasi topik berita. BERTopic hanya di-*fit* pada subset Train untuk mencegah kebocoran informasi (*data leakage*), lalu mentransformasikan seluruh berita ke dalam 5 probabilitas topik utama.""")

add_code("""# Ekstraksi Topik menggunakan BERTopic
tr_filter = df_raw[date_col] <= train_end
train_docs = (df_raw.loc[tr_filter, "title_clean"] + " " + df_raw.loc[tr_filter, "fulltext_clean"]).tolist()
all_docs   = (df_raw["title_clean"] + " " + df_raw["fulltext_clean"]).tolist()

print("Memulai ekstraksi fitur topik...")
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    topic_model = BERTopic(embedding_model=st_model, nr_topics=5, verbose=False, random_state=SEED)
    topic_model.fit(train_docs)
    
    _, probs = topic_model.transform(all_docs)
    if probs is None or len(probs.shape) == 1:
        probs = np.zeros((len(all_docs), 5))
    elif probs.shape[1] < 5:
        probs = np.pad(probs, ((0, 0), (0, 5 - probs.shape[1])))
    else:
        probs = probs[:, :5]
    print("BERTopic berhasil mengekstrak 5 probabilitas topik.")
except Exception as e:
    print(f"Fallback N-Gram Topic Model (karena: {e}):")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    
    vec = TfidfVectorizer(max_features=500, stop_words="english")
    X_tr_tfidf = vec.fit_transform(train_docs)
    svd = TruncatedSVD(n_components=5, random_state=SEED)
    svd.fit(X_tr_tfidf)
    probs = svd.transform(vec.transform(all_docs))
    probs = np.clip(probs, 0, None)
    print("N-Gram SVD Topics berhasil dibuat (5 komponen).")

for k in range(5):
    df_raw[f"topic_prob_{k}"] = probs[:, k]
""")

# =====================================================================
# Cell 9 & 10: GPR Lexicon Features
# =====================================================================
add_md("""## 4. Ekstraksi Fitur Geopolitical Risk (GPR-Based Lexicon)
Mengukur intensitas risiko geopolitik berdasarkan leksikon standar *Caldara & Iacoviello (2022)* yang mencakup istilah perang, militer, sanksi, ancaman, dan krisis. Fitur yang dihasilkan:
- `gpr_count`: Total frekuensi kata geopolitik.
- `gpr_density`: Densitas kata geopolitik terhadap total kata.""")

add_code("""GPR_TERMS = [
    "war", "military", "conflict", "geopolitic", "geopolitical", "threat", "sanction", "sanctions",
    "tension", "tensions", "crisis", "attack", "invasion", "nuclear", "terrorism", "terrorist",
    "missile", "hostility", "clash", "rebel", "insurgency", "troops", "sovereignty", "escalation",
    "retaliation", "dispute", "standoff", "blockade", "weapon", "drills", "border", "combat",
    "perang", "militer", "konflik", "geopolitik", "ancaman", "sanksi", "ketegangan", "krisis",
    "serangan", "invasi", "nuklir", "terorisme", "teroris", "rudal", "bentrokan", "pemberontak",
    "pasukan", "kedaulatan", "eskalasi", "pembalasan", "sengketa", "senjata", "perbatasan"
]

def extract_gpr_features(title, body):
    combined = (str(title) + " " + str(body)).lower()
    tokens = combined.split()
    total_tokens = len(tokens)
    if total_tokens == 0:
        return 0, 0.0
    gpr_c = sum(1 for tok in tokens if any(term in tok for term in GPR_TERMS))
    return gpr_c, gpr_c / total_tokens

gpr_metrics = [extract_gpr_features(t, b) for t, b in zip(df_raw["title_clean"], df_raw["fulltext_clean"])]
df_raw["gpr_count"]   = [m[0] for m in gpr_metrics]
df_raw["gpr_density"] = [m[1] for m in gpr_metrics]

print("Contoh Fitur GPR (5 Baris Teratas):")
print(df_raw[["gpr_count", "gpr_density"]].head())
""")

# =====================================================================
# Cell 11 & 12: FinBERT Sentiment & Polarity
# =====================================================================
add_md("""## 5. Analisis Sentimen Finansial & Polaritas Menggunakan FinBERT
Mengekstrak sentimen finansial dari judul berita menggunakan `ProsusAI/finbert`. Menghasilkan:
- Probabilitas: $P(\\text{Positive})$, $P(\\text{Negative})$, $P(\\text{Neutral})$
- Skor Polaritas: $\\text{Polarity} = P(\\text{Positive}) - P(\\text{Negative}) \\in [-1.0, +1.0]$""")

add_code("""cache_sent_path = Path("data/processed/finbert_sentiment.csv")
use_cached_sent = cache_sent_path.exists()

if use_cached_sent:
    print(f"Memuat FinBERT sentiment dari cache lokal: {cache_sent_path}")
    sent_df = pd.read_csv(cache_sent_path)
    for col in ["sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_polarity"]:
        df_raw[col] = sent_df[col]
else:
    print("Menjalankan inferensi FinBERT...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    fin_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    fin_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL).to(DEVICE)
    fin_model.eval()
    
    pos_list, neg_list, neu_list = [], [], []
    batch_size = 32
    titles = df_raw["title_clean"].tolist()
    
    with torch.no_grad():
        for i in range(0, len(titles), batch_size):
            batch_text = titles[i:i + batch_size]
            inputs = fin_tokenizer(batch_text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
            logits = fin_model(**inputs).logits
            probs_t = torch.softmax(logits, dim=-1).cpu().numpy()
            
            # ProsusAI/finbert: 0->positive, 1->negative, 2->neutral
            pos_list.extend(probs_t[:, 0])
            neg_list.extend(probs_t[:, 1])
            neu_list.extend(probs_t[:, 2])
            
    df_raw["sentiment_pos"] = pos_list
    df_raw["sentiment_neg"] = neg_list
    df_raw["sentiment_neu"] = neu_list
    df_raw["sentiment_polarity"] = np.array(pos_list) - np.array(neg_list)
    
    cache_sent_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw[["sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_polarity"]].to_csv(cache_sent_path, index=False)
    print("Hasil inferensi FinBERT berhasil di-cache ke disk.")

print("Statistik Sentimen FinBERT:")
print(df_raw[["sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_polarity"]].describe().round(4))
""")

# =====================================================================
# Cell 13 & 14: Dual-Encoder Embedding Concat
# =====================================================================
add_md("""## 6. Ekstraksi Dual-Encoder Embedding (Title BERT + Full-Text Longformer Concat)
- Title di-embed menggunakan `bert-base-uncased` ($768$ dimensi).
- Full-text di-embed menggunakan `allenai/longformer-base-4096` ($768$ dimensi).
- Kedua vektor di-concat horizontal menjadi **1536 dimensi**, lalu diagregasi harian dengan *rolling window* 5 hari dan direduksi menggunakan PCA (16 komponen).""")

add_code("""cache_dir = Path("data/processed")
emb_concat_cache = cache_dir / "emb_text_concat.npy"

if emb_concat_cache.exists():
    print(f"Memuat embedding gabungan (1536-d) dari cache: {emb_concat_cache}")
    X_concat_all = np.load(emb_concat_cache)
else:
    print("Mengekstrak embedding teks menggunakan Dual-Encoder...")
    from transformers import AutoTokenizer, AutoModel
    
    def extract_cls(texts, model_name, max_len=128):
        tok = AutoTokenizer.from_pretrained(model_name)
        mod = AutoModel.from_pretrained(model_name).to(DEVICE)
        mod.eval()
        embs = []
        bs = 16
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                b_texts = texts[i:i + bs]
                inp = tok(b_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEVICE)
                out = mod(**inp)
                cls_vec = out.last_hidden_state[:, 0, :].cpu().numpy()
                embs.append(cls_vec)
        return np.vstack(embs)
    
    emb_title = extract_cls(df_raw["title_clean"].tolist(), TITLE_BERT_MODEL, max_len=128)
    try:
        emb_body = extract_cls(df_raw["fulltext_clean"].tolist(), FULLTEXT_MODEL, max_len=1024)
    except Exception as e:
        print(f"Fallback Longformer ke BERT (karena: {e})")
        emb_body = extract_cls(df_raw["fulltext_clean"].tolist(), TITLE_BERT_MODEL, max_len=512)
        
    X_concat_all = np.hstack([emb_title, emb_body])
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_concat_cache, X_concat_all)
    print(f"Dual-encoder embeddings berhasil disimpan: shape {X_concat_all.shape}")

# Agregasi Harian & Rolling 5 Hari
daily_emb = pd.DataFrame(X_concat_all).groupby(df_raw[date_col]).mean()
daily_emb_roll = daily_emb.rolling(5, min_periods=1).mean()

# PCA Reduksi ke 16 Komponen (fit di Train saja)
train_dates_emb = daily_emb_roll.index <= train_end
pca = PCA(n_components=16, random_state=SEED)
pca.fit(daily_emb_roll[train_dates_emb])
pca_features = pd.DataFrame(pca.transform(daily_emb_roll), index=daily_emb_roll.index, 
                            columns=[f"pca_{i}" for i in range(16)])
print(f"PCA Variance Explained (16 komponen): {pca.explained_variance_ratio_.sum():.4f}")
""")

# =====================================================================
# Cell 15 & 16: Feature Engineering Temporal Kurs
# =====================================================================
add_md("""## 7. Ekstraksi Fitur Temporal Kurs USD/IDR
Mengekstrak dinamika deret waktu nilai tukar:
- Lagged returns: $ret_{t-1}, ret_{t-2}, ret_{t-3}$
- Rolling Volatility: 5-hari dan 20-hari
- Price Momentum: 5-hari dan 20-hari
- Hari dalam seminggu (*Day of Week*)""")

add_code("""# Agregasi Fitur Teks ke Level Harian
text_daily_df = df_raw.groupby(date_col).agg({
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
}).join(pca_features)

# Hitung Return Harian dan Fitur Temporal
text_daily_df["ret"]      = text_daily_df["rate_clean"].pct_change()
text_daily_df["ret_lag1"] = text_daily_df["ret"].shift(1)
text_daily_df["ret_lag2"] = text_daily_df["ret"].shift(2)
text_daily_df["ret_lag3"] = text_daily_df["ret"].shift(3)
text_daily_df["vol_5d"]   = text_daily_df["ret"].rolling(5).std()
text_daily_df["vol_20d"]  = text_daily_df["ret"].rolling(20).std()
text_daily_df["mom_5d"]   = text_daily_df["rate_clean"] / text_daily_df["rate_clean"].shift(5) - 1.0
text_daily_df["mom_20d"]  = text_daily_df["rate_clean"] / text_daily_df["rate_clean"].shift(20) - 1.0
text_daily_df["dow"]      = text_daily_df.index.dayofweek

# Target Pergerakan Kurs
text_daily_df["ret_magnitude"] = text_daily_df["ret"].abs()
text_daily_df["ret_direction"] = (text_daily_df["ret"] > 0).astype(int) # 1 = Melemah/Naik, 0 = Menguat/Turun

# Filter data baris yang memiliki nilai lengkap
daily_valid = text_daily_df.dropna(subset=["ret", "ret_lag3", "vol_20d", "mom_20d"]).copy()
print(f"Total baris valid harian: {len(daily_valid)}")
""")

# =====================================================================
# Cell 17 & 18: Stage 1a & Stage 1b (Magnitude & Direction Classifier)
# =====================================================================
add_md("""## 8. Pemodelan Tahap 1: Ekstraksi Sinyal Teks Terpadu
Untuk mengatasi kelemahan model sebelumnya di mana magnitude tidak memiliki tanda arah (*unidirectional*), kita melatih **dua model sinyal teks**:
1. **Stage 1a (Text-MLP Regressor)**: Memprediksi besarnya pergerakan pasar ($|ret_t|$) untuk menangkap intensitas volatilitas.
2. **Stage 1b (Text Direction Classifier)**: Memprediksi probabilitas kurs naik/melemah ($P(\\text{Up}) = P(ret_t > 0)$) dari sentimen, topik, dan risiko geopolitik.
3. **Signed Text Signal**: Menggabungkan probabilitas arah dan skala besaran:
$$\\text{Signed Signal}_t = (2 \\cdot P(\\text{Up}) - 1) \\times \\text{Magnitude}_t \\in [-\\text{Mag}, +\\text{Mag}]$$""")

add_code("""# Mask Pembagian Data Kronologis
tr_m = daily_valid.index <= train_end
va_m = (daily_valid.index > train_end) & (daily_valid.index <= val_end)
te_m = daily_valid.index > val_end

text_feat_cols = [
    "gpr_count", "gpr_density", "sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_polarity",
    "topic_prob_0", "topic_prob_1", "topic_prob_2", "topic_prob_3", "topic_prob_4"
] + [f"pca_{i}" for i in range(16)]

scaler_text_stage1 = StandardScaler().fit(daily_valid.loc[tr_m, text_feat_cols])
X_txt_tr = scaler_text_stage1.transform(daily_valid.loc[tr_m, text_feat_cols])
X_txt_va = scaler_text_stage1.transform(daily_valid.loc[va_m, text_feat_cols])
X_txt_te = scaler_text_stage1.transform(daily_valid.loc[te_m, text_feat_cols])
X_txt_all = scaler_text_stage1.transform(daily_valid[text_feat_cols])

# ---------------------------------------------------------------------
# A. Stage 1a: Text-MLP Regressor (Target: Magnitude |ret|)
# ---------------------------------------------------------------------
y_mag_tr = daily_valid.loc[tr_m, "ret_magnitude"].values
y_mag_te = daily_valid.loc[te_m, "ret_magnitude"].values

mlp_mag = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", alpha=0.1,
                       learning_rate_init=0.001, max_iter=250, random_state=SEED)
mlp_mag.fit(X_txt_tr, y_mag_tr)

daily_valid["pred_text_magnitude"] = mlp_mag.predict(X_txt_all)
mag_mae_te = mean_absolute_error(y_mag_te, daily_valid.loc[te_m, "pred_text_magnitude"])
print(f"Stage 1a Text-MLP Regressor  (Magnitude MAE Test Set): {mag_mae_te:.5f}")

# ---------------------------------------------------------------------
# B. Stage 1b: Text Direction Classifier (Target: ret > 0)
# ---------------------------------------------------------------------
y_dir_tr = daily_valid.loc[tr_m, "ret_direction"].values
y_dir_te = daily_valid.loc[te_m, "ret_direction"].values

text_clf = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", alpha=0.5,
                         learning_rate_init=0.001, max_iter=250, random_state=SEED,
                         early_stopping=True, validation_fraction=0.15)
text_clf.fit(X_txt_tr, y_dir_tr)

# Probabilitas Rupiah Melemah / USD Menguat
daily_valid["pred_text_prob_up"] = text_clf.predict_proba(X_txt_all)[:, 1]

# Sinyal Bertanda (-Mag s/d +Mag)
daily_valid["pred_text_signed_signal"] = (2 * daily_valid["pred_text_prob_up"] - 1.0) * daily_valid["pred_text_magnitude"]

dir_acc_stage1 = accuracy_score(y_dir_te, (daily_valid.loc[te_m, "pred_text_prob_up"] >= 0.5).astype(int)) * 100
print(f"Stage 1b Text Direction Classifier (Akurasi Arah Test Set): {dir_acc_stage1:.2f}%")
""")

# =====================================================================
# Cell 19 & 20: Master Dataset & Feature Groups
# =====================================================================
add_md("""## 9. Penggabungan Master Dataset & Definisi Grup Fitur Ablasi
Menggabungkan sinyal teks hasil Tahap 1 dengan fitur temporal deret waktu kurs.""")

add_code("""GROUPS = {
    "temporal": ["ret_lag1", "ret_lag2", "ret_lag3", "vol_5d", "vol_20d", "mom_5d", "mom_20d", "dow"],
    "text_magnitude": ["pred_text_magnitude"],
    "text_direction": ["pred_text_prob_up", "pred_text_signed_signal"],
    "topik": [f"topic_prob_{i}" for i in range(5)],
    "gpr": ["gpr_count", "gpr_density"],
    "sentimen": ["sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_polarity"],
    "embedding_pca": [f"pca_{i}" for i in range(16)]
}

all_feature_cols = sum(GROUPS.values(), [])
print(f"Total fitur di Master Dataset: {len(all_feature_cols)} fitur.")
""")

# =====================================================================
# Cell 21 & 22: Stage 2 Master Modeling & Target Scaling
# =====================================================================
add_md("""## 10. Pemodelan Tahap 2: Master Regressor & Evaluasi 4 Metrik
Menggunakan **Target Scaling** ($z$-score pada $y$ training) dan regularisasi optimal (L2 Weight Decay) untuk mencegah ledakan residual. Kita mengevaluasi:
- **Baseline (Prediksi 0 & Rata-rata Train)**
- **Master MLP Regressor** (dengan $\\alpha=1.0$ dan $\\alpha=2.0$)
- **Master Ridge Regressor** (Shrinkage linier)
- **Blended Ensemble** (MLP + Ridge)

Metrik yang dihitung:
1. **RMSE** (Target: $\\mathbf{0.002XXXX}$)
2. **MAE**
3. **MSE**
4. **Directional Accuracy (DirAcc %)**
5. **$R^2_{\\text{oos}}$**""")

add_code("""# Fungsi Evaluasi Metrik Komprehensif
def compute_metrics(y_true, y_pred, y_train_ref):
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    
    denom = np.sum((y_true - y_train_ref) ** 2)
    r2 = float(1 - np.sum(err ** 2) / denom) if denom > 0 else 0.0
    
    valid_sign = (y_true != 0) & (y_pred != 0)
    da = float(np.mean(np.sign(y_true[valid_sign]) == np.sign(y_pred[valid_sign])) * 100) if valid_sign.any() else 50.0
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "DirAcc (%)": da, "R2_oos": r2}

# Data Master
scaler_master = StandardScaler().fit(daily_valid.loc[tr_m, all_feature_cols])
X_tr = scaler_master.transform(daily_valid.loc[tr_m, all_feature_cols])
X_va = scaler_master.transform(daily_valid.loc[va_m, all_feature_cols])
X_te = scaler_master.transform(daily_valid.loc[te_m, all_feature_cols])

y_tr = daily_valid.loc[tr_m, "ret"].values
y_va = daily_valid.loc[va_m, "ret"].values
y_te = daily_valid.loc[te_m, "ret"].values

mu_y, std_y = y_tr.mean(), y_tr.std()
y_tr_scaled = (y_tr - mu_y) / std_y
y_va_scaled = (y_va - mu_y) / std_y

# Model 1: Master MLP (alpha=1.0)
mlp_reg1 = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", solver="adam", alpha=1.0,
                        learning_rate_init=0.001, max_iter=300, random_state=SEED)
mlp_reg1.fit(X_tr, y_tr_scaled)
pred_mlp1_te = mlp_reg1.predict(X_te) * std_y + mu_y

# Model 2: Master MLP (alpha=2.0)
mlp_reg2 = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", solver="adam", alpha=2.0,
                        learning_rate_init=0.001, max_iter=300, random_state=SEED)
mlp_reg2.fit(X_tr, y_tr_scaled)
pred_mlp2_te = mlp_reg2.predict(X_te) * std_y + mu_y

# Model 3: Master Ridge (alpha=10.0)
ridge_reg = Ridge(alpha=10.0, random_state=SEED)
ridge_reg.fit(X_tr, y_tr_scaled)
pred_ridge_te = ridge_reg.predict(X_te) * std_y + mu_y

# Model 4: Blended Ensemble (50% MLP + 50% Ridge)
pred_blend_te = 0.5 * pred_mlp1_te + 0.5 * pred_ridge_te

# Kompilasi Evaluasi
results_list = [
    {"Model": "Baseline: Tebak Return 0",         **compute_metrics(y_te, np.zeros_like(y_te), mu_y)},
    {"Model": "Baseline: Tebak Rata-rata Train",  **compute_metrics(y_te, np.full_like(y_te, mu_y), mu_y)},
    {"Model": "Master MLP Regressor (alpha=1.0)", **compute_metrics(y_te, pred_mlp1_te, mu_y)},
    {"Model": "Master MLP Regressor (alpha=2.0)", **compute_metrics(y_te, pred_mlp2_te, mu_y)},
    {"Model": "Master Ridge Regressor (Shrinkage)",**compute_metrics(y_te, pred_ridge_te, mu_y)},
    {"Model": "Ensemble Blend (MLP + Ridge)",      **compute_metrics(y_te, pred_blend_te, mu_y)}
]

eval_df = pd.DataFrame(results_list).set_index("Model")
print("\\n=================================== HASIL EVALUASI MODEL (TEST SET) ===================================")
display(eval_df.round(6))
""")

# =====================================================================
# Cell 23 & 24: Ablation Study
# =====================================================================
add_md("""## 11. Studi Ablasi Bertahap
Membuktikan secara bertahap pengaruh penambahan tiap kelompok fitur:
1. Temporal Kurs Saja
2. + Sinyal Magnitude Teks (Tahap 1a)
3. + Sinyal Arah Teks / Direction Classifier (Tahap 1b)
4. + Topik BERTopic
5. + Leksikon Geopolitik (GPR)
6. + Sentimen FinBERT
7. Full Pipeline (+ Dual-Encoder Concat Embedding PCA)""")

add_code("""ablation_steps = [
    ("1. Temporal Kurs Saja", ["temporal"]),
    ("2. + Sinyal Magnitude Teks", ["temporal", "text_magnitude"]),
    ("3. + Sinyal Arah Teks (Classifier)", ["temporal", "text_magnitude", "text_direction"]),
    ("4. + Topik BERTopic", ["temporal", "text_magnitude", "text_direction", "topik"]),
    ("5. + Leksikon Geopolitik (GPR)", ["temporal", "text_magnitude", "text_direction", "topik", "gpr"]),
    ("6. + Sentimen FinBERT", ["temporal", "text_magnitude", "text_direction", "topik", "gpr", "sentimen"]),
    ("7. Full Multimodal (+ Dual-Encoder PCA)", ["temporal", "text_magnitude", "text_direction", "topik", "gpr", "sentimen", "embedding_pca"]),
]

ablation_rows = []
for label, group_keys in ablation_steps:
    selected_cols = sum((GROUPS[k] for k in group_keys), [])
    sc = StandardScaler().fit(daily_valid.loc[tr_m, selected_cols])
    X_tr_abl = sc.transform(daily_valid.loc[tr_m, selected_cols])
    X_te_abl = sc.transform(daily_valid.loc[te_m, selected_cols])
    
    m_abl = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", alpha=1.0, max_iter=300, random_state=SEED)
    m_abl.fit(X_tr_abl, y_tr_scaled)
    p_abl = m_abl.predict(X_te_abl) * std_y + mu_y
    
    ablation_rows.append({"Tahapan Fitur": label, **compute_metrics(y_te, p_abl, mu_y)})

ablation_df = pd.DataFrame(ablation_rows).set_index("Tahapan Fitur")
print("\\n=================================== HASIL STUDI ABLASI (TEST SET) ===================================")
display(ablation_df.round(6))
""")

# =====================================================================
# Cell 25 & 26: Visualization
# =====================================================================
add_md("""## 12. Visualisasi Komparasi Kinerja
Menampilkan:
1. Komparasi Bar Chart untuk **RMSE, MAE, dan MSE** pada Test Set.
2. Komparasi Bar Chart **Directional Accuracy (%)**.
3. Plot Deret Waktu Pergerakan Return Aktual vs Hasil Prediksi Model.""")

add_code("""fig, axes = plt.subplots(1, 3, figsize=(20, 5), dpi=150)

# Subplot 1: Bar Chart RMSE & MAE
eval_plot_df = eval_df.drop(["Baseline: Tebak Return 0", "Baseline: Tebak Rata-rata Train"])
x_pos = np.arange(len(eval_plot_df))
width = 0.35

axes[0].bar(x_pos - width/2, eval_plot_df["RMSE"], width, label="RMSE", color="#2a9d8f")
axes[0].bar(x_pos + width/2, eval_plot_df["MAE"], width, label="MAE", color="#e76f51")
axes[0].axhline(0.0029, color="red", linestyle="--", alpha=0.7, label="Baseline RMSE (~0.0029)")
axes[0].set_title("Perbandingan Error (RMSE & MAE) di Test Set", fontsize=11, fontweight="bold")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(eval_plot_df.index, rotation=35, ha="right", fontsize=9)
axes[0].set_ylabel("Nilai Error")
axes[0].legend(fontsize=9)
axes[0].grid(axis="y", linestyle=":", alpha=0.6)

# Subplot 2: Bar Chart Directional Accuracy
axes[1].bar(x_pos, eval_plot_df["DirAcc (%)"], color="#457b9d", width=0.5)
axes[1].axhline(50.0, color="gray", linestyle="--", label="Tebak Koin (50%)")
axes[1].set_title("Directional Accuracy (%) di Test Set", fontsize=11, fontweight="bold")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(eval_plot_df.index, rotation=35, ha="right", fontsize=9)
axes[1].set_ylabel("Akurasi Arah (%)")
axes[1].set_ylim(40, 85)
for i, v in enumerate(eval_plot_df["DirAcc (%)"]):
    axes[1].text(i, v + 1.0, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].grid(axis="y", linestyle=":", alpha=0.6)

# Subplot 3: Time Series Actual vs Predicted (Ensemble Blend)
test_dates = daily_valid.loc[te_m].index
axes[2].plot(test_dates, y_te, label="Return Aktual (y_true)", color="black", alpha=0.7, linewidth=1.2)
axes[2].plot(test_dates, pred_blend_te, label="Prediksi Ensemble Blend", color="#e63946", linestyle="--", linewidth=1.4)
axes[2].set_title("Deret Waktu Return Kurs: Aktual vs Prediksi", fontsize=11, fontweight="bold")
axes[2].set_ylabel("Daily Return (ret)")
axes[2].legend(fontsize=9)
axes[2].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()
""")

# =====================================================================
# Cell 27 & 28: Export Reports
# =====================================================================
add_md("""## 13. Simpan Hasil Evaluasi & Model Artifacts
Menyimpan ringkasan metrik evaluasi dan dataset master ke direktori `reports/`.""")

add_code("""reports_dir = Path("percobaan task 3/reports")
if not reports_dir.exists():
    reports_dir = Path("reports")
reports_dir.mkdir(parents=True, exist_ok=True)

eval_csv_path = reports_dir / "evaluasi_rmse_002xxxx.csv"
ablation_csv_path = reports_dir / "ablation_results_direction_augmented.csv"
master_csv_path = reports_dir / "master_dataset_augmented.csv"

eval_df.to_csv(eval_csv_path)
ablation_df.to_csv(ablation_csv_path)
daily_valid.to_csv(master_csv_path)

print("Laporan dan artefak berhasil disimpan:")
print(f"  1. Ringkasan Evaluasi Model: {eval_csv_path}")
print(f"  2. Hasil Studi Ablasi      : {ablation_csv_path}")
print(f"  3. Master Dataset          : {master_csv_path}")
""")

# Build Notebook Object
nb_dict = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.11"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path_repo = "percobaan task 3/trial3_pipeline.ipynb"
with open(out_path_repo, "w", encoding="utf-8") as f:
    json.dump(nb_dict, f, indent=1, ensure_ascii=False)
print(f"Berhasil menulis ke: {out_path_repo}")

out_path_dl = "d:/Downloads/trial3_pipeline.ipynb"
with open(out_path_dl, "w", encoding="utf-8") as f:
    json.dump(nb_dict, f, indent=1, ensure_ascii=False)
print(f"Berhasil menulis ke: {out_path_dl}")
