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
### Arsitektur Valid Bebas Kebocoran: Out-of-Fold (OOF) Stacking, Selective Multimodal Features & Multi-Model Benchmark

Dokumen ini memuat implementasi pipeline terintegrasi hasil validasi metodologi:
1. **Pencegahan Data Leakage (Strict Zero Lookahead)**: Seluruh fitur deret waktu kurs dihitung murni berbasis informasi $t-1$ (`.shift(1)`).
2. **Ekstraksi Teks Pre-trained**: Dual-Encoder **BERT** (Headline/Title) + **Longformer** (Full-Text) diconcat ($1536$-d) dan diseleksi via PCA.
3. **Ekstraksi Fitur Tambahan**: Topik semantik (**BERTopic** terstandar dengan UMAP-HDBSCAN), **FinBERT** sentimen finansial & polaritas, serta leksikon risiko geopolitik (**GPR**).
4. **Tahap 1 (Stage 1) dengan Out-of-Fold (OOF) Stacking**: Memprediksi besaran gejolak ($|ret_t|$) dan probabilitas arah ($P(ret_t > 0)$) menggunakan 5-Fold OOF pada data latih guna mengeliminasi *in-sample stacking bias*.
5. **Tahap 2 (Stage 2) Multi-Model Benchmark**: Membandingkan **Deep MLP Regressor**, **Ridge Regressor**, **Gradient Boosting (GBDT)**, dan **Ensemble Blend (MLP + Ridge)** dengan *Target Scaling*.
6. **Evaluasi Finansial & Diagnostik**: Evaluasi **RMSE, MAE, MSE, Directional Accuracy (%)**, kurva *Learning Loss per Epoch*, serta analisis *Feature Importance*.
""")

# =====================================================================
# Cell 1 & 2: Install Dependencies
# =====================================================================
add_md("""## 0. Instalasi Library (Jika Diperlukan)
Jalankan sel ini apabila dependensi belum terinstal di environment lokal.""")

add_code("""# Instalasi library pendukung
!pip install -q sentence-transformers bertopic umap-learn hdbscan transformers torch scikit-learn
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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
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
print(f"Device aktif       : {DEVICE}")
print(f"Pretrained Title   : {TITLE_BERT_MODEL}")
print(f"Pretrained Fulltext: {FULLTEXT_MODEL}")
print(f"Pretrained FinBERT : {FINBERT_MODEL}")
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
    Path("/kaggle/input/datasets/graceee12/process-dataset/processed/merged_data_clean.csv"),
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

# 1. Format tanggal terurut kronologis
date_col = "Tanggal_dt" if "Tanggal_dt" in df_raw.columns else ("tanggal" if "tanggal" in df_raw.columns else "Tanggal")
df_raw[date_col] = pd.to_datetime(df_raw[date_col])
df_raw = df_raw.sort_values(date_col).reset_index(drop=True)

# 2. Hitung Kurs Tengah Bank Indonesia
if "Kurs Jual" in df_raw.columns and "Kurs Beli" in df_raw.columns:
    df_raw["rate_clean"] = (df_raw["Kurs Jual"] + df_raw["Kurs Beli"]) / 2.0
elif "rate_clean" not in df_raw.columns and "rate" in df_raw.columns:
    df_raw["rate_clean"] = df_raw["rate"]

# 3. Pembersihan Teks
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
# Cell 7 & 8: BERTopic Extraction (Fixed)
# =====================================================================
add_md("""## 3. Ekstraksi Fitur Topik Semantik (BERTopic)
Mengekstraksi representasi topik berita. BERTopic hanya di-*fit* pada subset Train untuk mencegah kebocoran informasi (*data leakage*), lalu mentransformasikan seluruh berita ke dalam 5 probabilitas topik utama.""")

add_code("""cache_topic_path = Path("data/processed/bertopic_probabilities.csv")

tr_filter = df_raw[date_col] <= train_end
train_docs = (df_raw.loc[tr_filter, "title_clean"] + " " + df_raw.loc[tr_filter, "fulltext_clean"]).tolist()
all_docs   = (df_raw["title_clean"] + " " + df_raw["fulltext_clean"]).tolist()

if cache_topic_path.exists():
    print(f"Memuat probabilitas BERTopic dari cache lokal: {cache_topic_path}")
    probs = pd.read_csv(cache_topic_path).values
else:
    print("Memulai ekstraksi fitur topik menggunakan BERTopic...")
    train_docs_clean = [d if d.strip() else "berita pasar uang ekonomi kurs valuta asing" for d in train_docs]
    all_docs_clean   = [d if d.strip() else "berita pasar uang ekonomi kurs valuta asing" for d in all_docs]
    
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from umap import UMAP
        from hdbscan import HDBSCAN
        
        # 1. Konfigurasi UMAP & HDBSCAN dengan seed deterministik
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED)
        hdbscan_model = HDBSCAN(min_cluster_size=10, min_samples=5, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
        
        # 2. Embedding model
        st_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
        
        # 3. Inisialisasi BERTopic (calculate_probabilities=True)
        topic_model = BERTopic(
            embedding_model=st_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            nr_topics=5,
            calculate_probabilities=True,
            verbose=False
        )
        
        # 4. Fit pada Train set saja
        topic_model.fit(train_docs_clean)
        
        # 5. Transform seluruh dokumen
        _, probs = topic_model.transform(all_docs_clean)
        
        if probs is None or len(probs.shape) == 1:
            probs = np.zeros((len(all_docs), 5))
        elif probs.shape[1] < 5:
            probs = np.pad(probs, ((0, 0), (0, 5 - probs.shape[1])))
        else:
            probs = probs[:, :5]
            
        print("BERTopic berhasil mengekstrak 5 probabilitas topik!")
        cache_topic_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(probs, columns=[f"topic_prob_{k}" for k in range(5)]).to_csv(cache_topic_path, index=False)
        print(f"Probabilitas topik berhasil di-cache ke: {cache_topic_path}")
        
        try:
            topic_info = topic_model.get_topic_info()
            print("\\nRingkasan Topik BERTopic:")
            display(topic_info.head(6))
        except Exception:
            pass
            
    except Exception as e:
        print(f"Fallback N-Gram Topic Model (karena: {e}):")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        
        vec = TfidfVectorizer(max_features=500, stop_words="english")
        X_tr_tfidf = vec.fit_transform(train_docs_clean)
        svd = TruncatedSVD(n_components=5, random_state=SEED)
        svd.fit(X_tr_tfidf)
        probs = svd.transform(vec.transform(all_docs_clean))
        probs = np.clip(probs, 0, None)
        print("N-Gram SVD Topics berhasil dibuat (5 komponen).")

for k in range(5):
    df_raw[f"topic_prob_{k}"] = probs[:, k]
""")

# =====================================================================
# Cell 9 & 10: GPR Lexicon
# =====================================================================
add_md("""## 4. Ekstraksi Fitur Geopolitical Risk (GPR-Based Lexicon)
Mengukur intensitas risiko geopolitik berdasarkan leksikon standar *Caldara & Iacoviello (2022)*:
- `gpr_count`: Frekuensi kemunculan istilah geopolitik.
- `gpr_density`: Densitas kata geopolitik per artikel.""")

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
# Cell 11 & 12: FinBERT
# =====================================================================
add_md("""## 5. Analisis Sentimen Finansial & Polaritas Menggunakan FinBERT
Mengekstrak sentimen finansial dari judul berita menggunakan `ProsusAI/finbert`:
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
# Cell 13 & 14: Dual-Encoder Embedding & Selective PCA
# =====================================================================
add_md("""## 6. Ekstraksi Dual-Encoder Embedding & Selective PCA
- Title di-embed menggunakan `bert-base-uncased` ($768$ dimensi).
- Full-text di-embed menggunakan `allenai/longformer-base-4096` ($768$ dimensi).
- Kedua vektor diconcat horizontal ($1536$ dimensi), diagregasi harian dengan *rolling window* 5 hari, dan direduksi secara selektif ke **6 komponen utama PCA** guna mencegah *curse of dimensionality*.""")

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

# Selective PCA: 6 Komponen Utama (fit di Train saja)
train_dates_emb = daily_emb_roll.index <= train_end
pca = PCA(n_components=6, random_state=SEED)
pca.fit(daily_emb_roll[train_dates_emb])
pca_features = pd.DataFrame(pca.transform(daily_emb_roll), index=daily_emb_roll.index, 
                            columns=[f"pca_{i}" for i in range(6)])
print(f"PCA Variance Explained (6 komponen utama): {pca.explained_variance_ratio_.sum():.4f}")
""")

# =====================================================================
# Cell 15 & 16: Clean Temporal Features (Zero Lookahead)
# =====================================================================
add_md("""## 7. Ekstraksi Fitur Temporal Kurs (Strict Zero Lookahead)
Seluruh fitur deret waktu kurs dihitung **murni berbasis informasi $t-1$** (`.shift(1)`), memastikan tidak ada kebocoran harga saat ini (*no lookahead bias*):
- Lagged returns: $ret_{t-1}, ret_{t-2}, ret_{t-3}$
- Rolling Volatility: 5-hari dan 20-hari (dihitung hingga $t-1$)
- Price Momentum: 5-hari dan 20-hari (dihitung hingga $t-1$)
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

# Target Prediksi Hari t
text_daily_df["ret"]           = text_daily_df["rate_clean"].pct_change()
text_daily_df["ret_magnitude"] = text_daily_df["ret"].abs()
text_daily_df["ret_direction"] = (text_daily_df["ret"] > 0).astype(int) # 1 = Melemah/Naik, 0 = Menguat/Turun

# FITUR DERET WAKTU BEBAS KEBOCORAN (Strictly Lagged t-1)
text_daily_df["ret_lag1"] = text_daily_df["ret"].shift(1)
text_daily_df["ret_lag2"] = text_daily_df["ret"].shift(2)
text_daily_df["ret_lag3"] = text_daily_df["ret"].shift(3)
text_daily_df["vol_5d"]   = text_daily_df["ret"].shift(1).rolling(5).std()
text_daily_df["vol_20d"]  = text_daily_df["ret"].shift(1).rolling(20).std()
text_daily_df["mom_5d"]   = text_daily_df["rate_clean"].shift(1) / text_daily_df["rate_clean"].shift(6) - 1.0
text_daily_df["mom_20d"]  = text_daily_df["rate_clean"].shift(1) / text_daily_df["rate_clean"].shift(21) - 1.0
text_daily_df["dow"]      = text_daily_df.index.dayofweek

daily_valid = text_daily_df.dropna(subset=["ret", "ret_lag3", "vol_20d", "mom_20d"]).copy()
print(f"Total baris valid harian bebas kebocoran: {len(daily_valid)}")
""")

# =====================================================================
# Cell 17 & 18: Stage 1 with Out-of-Fold (OOF) Stacking
# =====================================================================
add_md("""## 8. Pemodelan Tahap 1: Ekstraksi Sinyal Teks dengan Out-of-Fold (OOF) Stacking
Untuk mengeliminasi *in-sample stacking bias*, sinyal teks pada set pelatihan di-generate melalui **5-Fold Cross-Validation (Out-of-Fold)**:
1. **Stage 1a (Text-MLP Regressor)**: Memprediksi besarnya gerakan ($|ret_t|$).
2. **Stage 1b (Text Direction Classifier)**: Memprediksi probabilitas arah ($P(\\text{Up}) = P(ret_t > 0)$).
3. **Signed Text Signal**: $(2 \\cdot P(\\text{Up}) - 1) \\times \\text{Magnitude}_t$.""")

add_code("""# Mask Pembagian Data Kronologis
tr_m = daily_valid.index <= train_end
va_m = (daily_valid.index > train_end) & (daily_valid.index <= val_end)
te_m = daily_valid.index > val_end

text_feat_cols = [
    "gpr_count", "gpr_density", "sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_polarity",
    "topic_prob_0", "topic_prob_1", "topic_prob_2", "topic_prob_3", "topic_prob_4"
] + [f"pca_{i}" for i in range(6)]

scaler_txt_s1 = StandardScaler().fit(daily_valid.loc[tr_m, text_feat_cols])
X_txt_tr = scaler_txt_s1.transform(daily_valid.loc[tr_m, text_feat_cols])
X_txt_va = scaler_txt_s1.transform(daily_valid.loc[va_m, text_feat_cols])
X_txt_te = scaler_txt_s1.transform(daily_valid.loc[te_m, text_feat_cols])

y_mag_tr = daily_valid.loc[tr_m, "ret_magnitude"].values
y_dir_tr = daily_valid.loc[tr_m, "ret_direction"].values

# 5-Fold Out-of-Fold (OOF) pada Training Set
kf = KFold(n_splits=5, shuffle=False)
oof_pred_mag  = np.zeros(len(X_txt_tr))
oof_pred_prob = np.zeros(len(X_txt_tr))

for fold_tr, fold_val in kf.split(X_txt_tr):
    # Model Magnitude
    m_mag_fold = MLPRegressor(hidden_layer_sizes=(16,), alpha=1.0, max_iter=200, random_state=SEED)
    m_mag_fold.fit(X_txt_tr[fold_tr], y_mag_tr[fold_tr])
    oof_pred_mag[fold_val] = m_mag_fold.predict(X_txt_tr[fold_val])
    
    # Model Direction
    m_clf_fold = MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0, max_iter=200, random_state=SEED)
    m_clf_fold.fit(X_txt_tr[fold_tr], y_dir_tr[fold_tr])
    oof_pred_prob[fold_val] = m_clf_fold.predict_proba(X_txt_tr[fold_val])[:, 1]

# Model Penuh Fit di seluruh Train Set untuk Inferensi Validation & Test
m_mag_full = MLPRegressor(hidden_layer_sizes=(16,), alpha=1.0, max_iter=200, random_state=SEED).fit(X_txt_tr, y_mag_tr)
m_clf_full = MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0, max_iter=200, random_state=SEED).fit(X_txt_tr, y_dir_tr)

# Simpan Sinyal Teks OOF ke DataFrame
daily_valid.loc[tr_m, "pred_text_magnitude"] = oof_pred_mag
daily_valid.loc[tr_m, "pred_text_prob_up"]   = oof_pred_prob
daily_valid.loc[tr_m, "pred_text_signed_signal"] = (2 * oof_pred_prob - 1.0) * oof_pred_mag

daily_valid.loc[va_m, "pred_text_magnitude"] = m_mag_full.predict(X_txt_va)
daily_valid.loc[va_m, "pred_text_prob_up"]   = m_clf_full.predict_proba(X_txt_va)[:, 1]
daily_valid.loc[va_m, "pred_text_signed_signal"] = (2 * daily_valid.loc[va_m, "pred_text_prob_up"] - 1.0) * daily_valid.loc[va_m, "pred_text_magnitude"]

daily_valid.loc[te_m, "pred_text_magnitude"] = m_mag_full.predict(X_txt_te)
daily_valid.loc[te_m, "pred_text_prob_up"]   = m_clf_full.predict_proba(X_txt_te)[:, 1]
daily_valid.loc[te_m, "pred_text_signed_signal"] = (2 * daily_valid.loc[te_m, "pred_text_prob_up"] - 1.0) * daily_valid.loc[te_m, "pred_text_magnitude"]

test_dir_acc = accuracy_score(daily_valid.loc[te_m, "ret_direction"], (daily_valid.loc[te_m, "pred_text_prob_up"] >= 0.5).astype(int)) * 100
print(f"Stage 1b Text Direction Classifier (Akurasi Arah Test Set): {test_dir_acc:.2f}%")
print("OOF Stacking sinyal teks selesai dibuat.")
""")

# =====================================================================
# Cell 19 & 20: Master Dataset & Feature Groups
# =====================================================================
add_md("""## 9. Penggabungan Master Dataset & Definisi Grup Fitur""")

add_code("""GROUPS = {
    "temporal": ["ret_lag1", "ret_lag2", "ret_lag3", "vol_5d", "vol_20d", "mom_5d", "mom_20d", "dow"],
    "text_signals": ["pred_text_magnitude", "pred_text_prob_up", "pred_text_signed_signal"],
    "topik": [f"topic_prob_{i}" for i in range(5)],
    "gpr": ["gpr_count", "gpr_density"],
    "sentimen": ["sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_polarity"],
    "selective_pca": [f"pca_{i}" for i in range(6)]
}

all_feature_cols = sum(GROUPS.values(), [])
print(f"Total fitur di Master Dataset: {len(all_feature_cols)} fitur.")
""")

# =====================================================================
# Cell 21 & 22: Stage 2 Multi-Model Benchmark
# =====================================================================
add_md("""## 10. Pemodelan Tahap 2: Multi-Model Benchmark & Evaluasi Komprehensif
Menggunakan **Target Scaling** ($z$-score pada $y_{\\text{train}}$) dan regularisasi optimal. Mengevaluasi:
- **Baseline 0 & Baseline Rata-rata Train**
- **Master Ridge Regressor (Shrinkage)**
- **Master Deep MLP Regressor** (dengan pelacakan kurva konvergensi per epoch)
- **Gradient Boosting Regressor (GBDT)**
- **Ensemble Blend (MLP + Ridge)**""")

add_code("""def compute_metrics(y_true, y_pred, y_train_ref):
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    
    denom = np.sum((y_true - y_train_ref) ** 2)
    r2 = float(1 - np.sum(err ** 2) / denom) if denom > 0 else 0.0
    
    valid_sign = (y_true != 0) & (y_pred != 0)
    da = float(np.mean(np.sign(y_true[valid_sign]) == np.sign(y_pred[valid_sign])) * 100) if valid_sign.any() else 50.0
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "DirAcc (%)": da, "R2_oos": r2}

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

# 1. Master Ridge
ridge_model = Ridge(alpha=50.0, random_state=SEED).fit(X_tr, y_tr_scaled)
pred_ridge_te = ridge_model.predict(X_te) * std_y + mu_y

# 2. Master Deep MLP dengan Pelacakan Loss
mlp_model = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", solver="adam", alpha=2.0,
                         learning_rate_init=0.001, batch_size=32, random_state=SEED)

train_loss_hist = []
val_loss_hist   = []
best_loss, best_weights, wait_cnt = float("inf"), None, 0

for epoch in range(250):
    mlp_model.partial_fit(X_tr, y_tr_scaled)
    tr_loss = np.mean((mlp_model.predict(X_tr) - y_tr_scaled)**2)
    va_loss = np.mean((mlp_model.predict(X_va) - y_va_scaled)**2)
    train_loss_hist.append(tr_loss)
    val_loss_hist.append(va_loss)
    
    if va_loss < best_loss - 1e-5:
        best_loss = va_loss
        best_weights = (copy.deepcopy(mlp_model.coefs_), copy.deepcopy(mlp_model.intercepts_))
        wait_cnt = 0
    else:
        wait_cnt += 1
        if wait_cnt >= 30:
            break

if best_weights is not None:
    mlp_model.coefs_, mlp_model.intercepts_ = best_weights

pred_mlp_te = mlp_model.predict(X_te) * std_y + mu_y

# 3. Gradient Boosting Regressor (GBDT)
gbr_model = GradientBoostingRegressor(n_estimators=70, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=SEED)
gbr_model.fit(X_tr, y_tr_scaled)
pred_gbr_te = gbr_model.predict(X_te) * std_y + mu_y

# 4. Ensemble Blend (50% MLP + 50% Ridge)
pred_blend_te = 0.5 * pred_mlp_te + 0.5 * pred_ridge_te

# Tabel Evaluasi Multi-Model
results_list = [
    {"Model": "Baseline: Tebak Return 0",           **compute_metrics(y_te, np.zeros_like(y_te), mu_y)},
    {"Model": "Baseline: Tebak Rata-rata Train",    **compute_metrics(y_te, np.full_like(y_te, mu_y), mu_y)},
    {"Model": "Master Ridge Regressor (Shrinkage)", **compute_metrics(y_te, pred_ridge_te, mu_y)},
    {"Model": "Master Deep MLP Regressor",          **compute_metrics(y_te, pred_mlp_te, mu_y)},
    {"Model": "Gradient Boosting (GBDT)",           **compute_metrics(y_te, pred_gbr_te, mu_y)},
    {"Model": "Ensemble Blend (MLP + Ridge)",        **compute_metrics(y_te, pred_blend_te, mu_y)}
]

eval_df = pd.DataFrame(results_list).set_index("Model")
print("\\n=================================== HASIL EVALUASI MULTI-MODEL (TEST SET) ===================================")
display(eval_df.round(6))
""")

# =====================================================================
# Cell 23 & 24: Ablation Study
# =====================================================================
add_md("""## 11. Studi Ablasi Bertahap (Bebas Kebocoran)""")

add_code("""ablation_steps = [
    ("1. Temporal Kurs Saja (Lagged t-1)", ["temporal"]),
    ("2. + Sinyal Teks OOF (Magnitude & Arah)", ["temporal", "text_signals"]),
    ("3. + Topik BERTopic", ["temporal", "text_signals", "topik"]),
    ("4. + Leksikon Geopolitik (GPR)", ["temporal", "text_signals", "topik", "gpr"]),
    ("5. + Sentimen FinBERT", ["temporal", "text_signals", "topik", "gpr", "sentimen"]),
    ("6. Full Multimodal (+ Selective PCA)", ["temporal", "text_signals", "topik", "gpr", "sentimen", "selective_pca"]),
]

ablation_rows = []
for label, group_keys in ablation_steps:
    selected_cols = sum((GROUPS[k] for k in group_keys), [])
    sc = StandardScaler().fit(daily_valid.loc[tr_m, selected_cols])
    X_tr_abl = sc.transform(daily_valid.loc[tr_m, selected_cols])
    X_te_abl = sc.transform(daily_valid.loc[te_m, selected_cols])
    
    m_abl = Ridge(alpha=50.0, random_state=SEED).fit(X_tr_abl, y_tr_scaled)
    p_abl = m_abl.predict(X_te_abl) * std_y + mu_y
    
    ablation_rows.append({"Tahapan Fitur": label, **compute_metrics(y_te, p_abl, mu_y)})

ablation_df = pd.DataFrame(ablation_rows).set_index("Tahapan Fitur")
print("\\n=================================== HASIL STUDI ABLASI (TEST SET) ===================================")
display(ablation_df.round(6))
""")

# =====================================================================
# Cell 25 & 26: 5-Panel Visualization
# =====================================================================
add_md("""## 12. Visualisasi Komprehensif (Metrik, Deret Waktu, Kurva Loss, & Feature Importance)
Menampilkan 5 panel visualisasi:
1. Bar Chart **RMSE & MAE**
2. Bar Chart **Directional Accuracy (%)**
3. Time Series **Return Aktual vs Prediksi**
4. Kurva **Learning Loss per Epoch (Train vs Val)** untuk Deep MLP
5. Koefisien Pengaruh Fitur (**Feature Importance** Ridge Regressor)""")

add_code("""fig = plt.figure(figsize=(20, 10), dpi=150)
gs = fig.add_gridspec(2, 6, hspace=0.35, wspace=0.35)

eval_plot = eval_df.drop(["Baseline: Tebak Return 0", "Baseline: Tebak Rata-rata Train"])
x_idx = np.arange(len(eval_plot))
w = 0.35

# 1. RMSE & MAE Bar Chart
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.bar(x_idx - w/2, eval_plot["RMSE"], w, label="RMSE", color="#2a9d8f")
ax1.bar(x_idx + w/2, eval_plot["MAE"], w, label="MAE", color="#e76f51")
ax1.axhline(0.0029, color="red", linestyle="--", alpha=0.7, label="Baseline (~0.0029)")
ax1.set_title("Perbandingan Error (RMSE & MAE) Test Set", fontweight="bold", fontsize=11)
ax1.set_xticks(x_idx)
ax1.set_xticklabels(eval_plot.index, rotation=25, ha="right", fontsize=9)
ax1.set_ylabel("Nilai Error")
ax1.legend(fontsize=9)
ax1.grid(axis="y", linestyle=":", alpha=0.6)

# 2. Directional Accuracy Bar Chart
ax2 = fig.add_subplot(gs[0, 2:4])
ax2.bar(x_idx, eval_plot["DirAcc (%)"], color="#457b9d", width=0.5)
ax2.axhline(50.0, color="gray", linestyle="--", label="Tebak Koin (50%)")
ax2.set_title("Directional Accuracy (%) Test Set", fontweight="bold", fontsize=11)
ax2.set_xticks(x_idx)
ax2.set_xticklabels(eval_plot.index, rotation=25, ha="right", fontsize=9)
ax2.set_ylabel("Akurasi Arah (%)")
ax2.set_ylim(45, 65)
for i, v in enumerate(eval_plot["DirAcc (%)"]):
    ax2.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(axis="y", linestyle=":", alpha=0.6)

# 3. Time Series Plot
ax3 = fig.add_subplot(gs[0, 4:6])
test_dates = daily_valid.loc[te_m].index
ax3.plot(test_dates, y_te, label="Return Aktual", color="black", alpha=0.7, linewidth=1.2)
ax3.plot(test_dates, pred_blend_te, label="Prediksi Ensemble", color="#e63946", linestyle="--", linewidth=1.3)
ax3.set_title("Deret Waktu Return: Aktual vs Prediksi", fontweight="bold", fontsize=11)
ax3.set_ylabel("Return (ret)")
ax3.legend(fontsize=9)
ax3.grid(True, linestyle=":", alpha=0.6)

# 4. Kurva Konvergensi MLP Loss per Epoch
ax4 = fig.add_subplot(gs[1, 0:3])
ax4.plot(train_loss_hist, label="Training Loss (MSE)", color="#1d3557", linewidth=1.5)
ax4.plot(val_loss_hist, label="Validation Loss (MSE)", color="#e63946", linestyle="--", linewidth=1.5)
ax4.set_title("Kurva Pembelajaran Master Deep MLP (Loss vs Epoch)", fontweight="bold", fontsize=11)
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Scaled MSE Loss")
ax4.legend(fontsize=9)
ax4.grid(True, linestyle=":", alpha=0.6)

# 5. Feature Importance (Koefisien Ridge)
ax5 = fig.add_subplot(gs[1, 3:6])
coef_series = pd.Series(ridge_model.coef_, index=all_feature_cols)
top_coefs = pd.concat([coef_series.nlargest(6), coef_series.nsmallest(6)]).drop_duplicates()
top_coefs = top_coefs.sort_values()

colors = ["#e76f51" if c < 0 else "#2a9d8f" for c in top_coefs.values]
ax5.barh(top_coefs.index, top_coefs.values, color=colors)
ax5.axvline(0, color="black", linewidth=0.8)
ax5.set_title("Pengaruh Fitur Utama Terhadap Kurs (Ridge Coefficients)", fontweight="bold", fontsize=11)
ax5.set_xlabel("Nilai Koefisien Regresi Terstandar")
ax5.grid(axis="x", linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()
""")

# =====================================================================
# Cell 27 & 28: Export Reports
# =====================================================================
add_md("""## 13. Simpan Hasil Evaluasi & Model Artifacts
Menyimpan ringkasan metrik evaluasi dan dataset master ke direktori `reports/`.""")

add_code("""reports_dir = Path("percobaan task 3/reports")
reports_dir.mkdir(parents=True, exist_ok=True)

eval_csv_path = reports_dir / "evaluasi_multimodel_clean.csv"
ablation_csv_path = reports_dir / "ablation_results_clean.csv"
master_csv_path = reports_dir / "master_dataset_clean.csv"

eval_df.to_csv(eval_csv_path)
ablation_df.to_csv(ablation_csv_path)
daily_valid.to_csv(master_csv_path)

print("Laporan hasil eksperimen bebas kebocoran berhasil disimpan:")
print(f"  1. Ringkasan Evaluasi Multi-Model: {eval_csv_path}")
print(f"  2. Hasil Studi Ablasi           : {ablation_csv_path}")
print(f"  3. Master Dataset Bersih        : {master_csv_path}")
""")

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

# Tulis ke repo dan scratch
with open("percobaan task 3/trial3_pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb_dict, f, indent=1, ensure_ascii=False)
print("Berhasil menulis ke percobaan task 3/trial3_pipeline.ipynb")

with open("scratch/trial3_pipeline_clean.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb_dict, f, indent=1, ensure_ascii=False)
print("Berhasil menulis ke scratch/trial3_pipeline_clean.ipynb")
