import json
from pathlib import Path

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(code):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    })

# ----------------- CELL 0: TITLE & SUMMARY -----------------
add_md(r"""# Pipeline Eksperimen Task 3: Directional Entity-Based Multimodal USD/IDR Forecasting
### Arsitektur Klasifikasi Arah (USD vs IDR) & Prediksi Log Return Bebas Kebocoran

Notebook ini mengimplementasikan arsitektur pemodelan valas USD/IDR multimodal yang menyatukan analisis entitas berita per artikel, polaritas bipolar, klasifikasi arah *Out-of-Fold* (OOF), dan fitur deret waktu ringkas:
1. **Ekstraksi Per Artikel (`split(" | ")`)**: Memecah agregat harian menjadi artikel individual agar tidak terjadi pemotongan *token* (`truncation`) di FinBERT dan mencegah pencampuran entitas antar berita.
2. **Target-Based Sentiment Analysis (Metode 2)**:
   - Flag entitas **Amerika Serikat / USD** (`flag_us`, peka huruf besar untuk `U.S./US/USD/Fed` agar tidak salah mencocokkan kata ganti `"us"`) dan **Indonesia / Rupiah** (`flag_idr`).
   - Pembobotan sentimen: Sentimen positif pada AS mendorong penguatan USD (USD/IDR naik), sentimen positif pada Indonesia mendorong penguatan Rupiah (USD/IDR turun).
   - Polaritas FinBERT dipastikan **strictly bipolar** (hanya positif vs negatif tanpa dilusi netral: `pol_bin` $\in \{-1, +1\}$ dan `pol_margin` $= \frac{pos - neg}{pos + neg} \in [-1, +1]$). Hari tanpa berita diberi nilai $0$.
3. **Interaksi Geopolitical Risk (GPR)**:
   - Mengukur densitas risiko geopolitik dan interaksinya dengan polaritas negatif (`gpr_weak_margin`) sebagai sinyal *safe-haven* ke Dolar AS (pelemahan Rupiah).
4. **Penyelarasan Zona Waktu US $\to$ WIB & Stage 1 Klasifikasi Arah (USD vs IDR)**:
   - Memprediksi siapa yang naik ($1 = \text{USD Naik / Rupiah Melemah}$, $0 = \text{IDR Naik / Rupiah Menguat}$) menggunakan **Gradient Boosting Classifier** dengan **5-Fold OOF** pada berita yang diselaraskan ke hari perdagangan ($t-1 \to t$).
5. **Fitur Deret Waktu Ringkas (*Strict Zero Lookahead $t-1$*)**:
   - Dasar: `ret_lag1`, `ret_lag2`, `vol_5d`.
   - Indikator Reversal & Momentum: `consec_up_days` (hari naik/turun berturut-turut) dan `dist_ma20` ($\frac{\text{rate}_{t-1}}{\text{MA20}_{t-1}} - 1$).
6. **Stage 2: Prediksi `log_return` dengan Direction-Aware PyTorch MLP & Ridge**:
   - Menggunakan fungsi rugi komposit $\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{dir}}\mathcal{L}_{\text{sign}}$ agar prediksi tidak menyusut ke konstanta positif, serta mengevaluasi **RMSE, MAE, MSE, DirAcc (%), dan Balanced Accuracy (%)**.""")

# ----------------- CELL 1 & 2: SETUP & INSTALLATION -----------------
add_md(r"""## 0. Instalasi Library (Jika Diperlukan)
Jalankan sel ini apabila dependensi belum terinstal di environment lokal.""")

add_code(r"""# Instalasi library pendukung
!pip install -q sentence-transformers bertopic umap-learn hdbscan transformers torch scikit-learn""")

# ----------------- CELL 3 & 4: CONFIGURATION & SEED -----------------
add_md(r"""## 1. Konfigurasi Lingkungan, Device, dan Parameter Eksperimen
Menetapkan seed deterministik, device (CUDA/CPU), serta model pretrained yang digunakan.""")

add_code(r"""import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, accuracy_score,
    balanced_accuracy_score, roc_auc_score, confusion_matrix
)

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
print(f"Pretrained FinBERT : {FINBERT_MODEL}")""")

# ----------------- CELL 5 & 6: DATA LOADING & SPLIT -----------------
add_md(r"""## 2. Pemuatan Dataset & Pemisahan Kronologis (Title vs Full-Text)
Membaca dataset gabungan, menghitung kurs tengah $Kurs = \frac{Jual + Beli}{2}$, serta menetapkan pemisahan kronologis 70% Train, 15% Validation, dan 15% Test.""")

add_code(r"""# Jalur dataset utama
DATA_CANDIDATES = [
    Path("data/processed/merged_data_clean.csv"),
    Path("../data/processed/merged_data_clean.csv"),
    Path("/kaggle/input/datasets/graceee12/process-dataset/processed/merged_data_clean.csv"),
    Path("merged_data_clean.csv")
]

DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), None)
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

# 3. Fungsi Pembersihan Teks Ringan (pertahankan kapitalisasi untuk deteksi entitas U.S./US)
def clean_text_fn(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    return re.sub(r"\s+", " ", text).strip()

title_col = "all_title" if "all_title" in df_raw.columns else "title"
body_col  = "all_full_text" if "all_full_text" in df_raw.columns else ("all_fulltext" if "all_fulltext" in df_raw.columns else "full_text")

df_raw["title_clean"]    = df_raw[title_col].fillna("").apply(clean_text_fn)
df_raw["fulltext_clean"] = df_raw[body_col].fillna("").apply(clean_text_fn)

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
print(f"Periode Test      : {unique_dates.iloc[val_end_idx + 1].date()} s/d {unique_dates.iloc[-1].date()} ({n_dates - val_end_idx - 1} hari)")""")

# ----------------- CELL 7 & 8: BERTopic -----------------
add_md(r"""## 3. Ekstraksi Fitur Topik Semantik (BERTopic)
Mengekstraksi representasi topik berita. BERTopic hanya di-*fit* pada subset Train untuk mencegah kebocoran informasi (*data leakage*), lalu mentransformasikan seluruh dokumen ke dalam 5 probabilitas topik utama.""")

add_code(r"""cache_topic_path = Path("data/processed/bertopic_probabilities.csv")

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
        
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED)
        hdbscan_model = HDBSCAN(min_cluster_size=10, min_samples=5, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
        st_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
        
        topic_model = BERTopic(
            embedding_model=st_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            nr_topics=5,
            calculate_probabilities=True,
            verbose=False
        )
        topic_model.fit(train_docs_clean)
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
    except Exception as e:
        print(f"Fallback N-Gram Topic Model (karena: {e}):")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        
        vec = TfidfVectorizer(max_features=500, stop_words="english")
        X_tr_tfidf = vec.fit_transform(train_docs_clean)
        svd = TruncatedSVD(n_components=5, random_state=SEED)
        svd.fit(X_tr_tfidf)
        probs = np.clip(svd.transform(vec.transform(all_docs_clean)), 0, None)
        print("N-Gram SVD Topics berhasil dibuat (5 komponen).")

for k in range(5):
    df_raw[f"topic_prob_{k}"] = probs[:, k]""")

# ----------------- CELL 9 & 10: PER-ARTICLE FINBERT -----------------
add_md(r"""## 4. Inferensi Sentimen Finansial Per Artikel (FinBERT)
Memecah judul gabungan harian menggunakan pemisah `' | '` agar setiap artikel dianalisis secara penuh oleh `ProsusAI/finbert` tanpa terpotong (`truncation`). Hasilnya dideduplikasi dan disimpan ke cache agar eksekusi berikutnya sangat cepat.""")

add_code(r"""fin_cache_path = Path("data/processed/finbert_unique_titles.csv")

# Kumpulkan seluruh judul artikel individual
all_individual_titles = []
for raw_t in df_raw[title_col].fillna(""):
    for part in str(raw_t).split(" | "):
        t_c = clean_text_fn(part)
        if t_c and t_c.lower() != "nan":
            all_individual_titles.append(t_c)

uniq_titles = sorted(list(set(all_individual_titles)), key=len)
print(f"Total kemunculan artikel: {len(all_individual_titles)} | Judul unik: {len(uniq_titles)}")

if fin_cache_path.exists():
    print(f"Memuat FinBERT skor per artikel dari cache: {fin_cache_path}")
    fin_df = pd.read_csv(fin_cache_path)
    fin_map = fin_df.set_index("title")[["finbert_pos", "finbert_neg", "finbert_neu"]].to_dict("index")
else:
    fin_map = {}

missing_titles = [t for t in uniq_titles if t not in fin_map]
if missing_titles:
    print(f"Menjalankan inferensi FinBERT untuk {len(missing_titles)} judul baru...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    fin_tok = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    fin_mod = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL).to(DEVICE)
    fin_mod.eval()
    
    bs = 64
    with torch.no_grad():
        for i in range(0, len(missing_titles), bs):
            b = missing_titles[i:i+bs]
            inp = fin_tok(b, padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
            probs_t = torch.softmax(fin_mod(**inp).logits, dim=-1).cpu().numpy()
            for title_str, p_arr in zip(b, probs_t):
                # ProsusAI/finbert: 0->positive, 1->negative, 2->neutral
                fin_map[title_str] = {
                    "finbert_pos": float(p_arr[0]),
                    "finbert_neg": float(p_arr[1]),
                    "finbert_neu": float(p_arr[2])
                }
    fin_cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(fin_map, orient="index").reset_index().rename(columns={"index": "title"}).to_csv(fin_cache_path, index=False)
    print(f"Hasil FinBERT per artikel disimpan ke: {fin_cache_path}")""")

# ----------------- CELL 11 & 12: TARGET-BASED ENTITY & BIPOLAR POLARITY + GPR -----------------
add_md(r"""## 5. Target-Based Entity Sentiment (US vs IDR), Polaritas Bipolar, & Interaksi GPR
Menghitung sinyal arah di level **per artikel** sebelum diagregasi ke level harian:
- **Flag Entitas (`flag_us`, `flag_idr`)**: Menggunakan regex peka huruf kapital agar akronim `U.S./US/USD/Fed` tidak tertukar dengan kata ganti bahasa Inggris `"us"`.
- **Polaritas Strictly Positif vs Negatif**:
  $$\text{pol\_bin} = \begin{cases} +1, & P(\text{Pos}) \ge P(\text{Neg}) \\ -1, & P(\text{Pos}) < P(\text{Neg}) \end{cases}, \qquad \text{pol\_margin} = \frac{P(\text{Pos}) - P(\text{Neg})}{P(\text{Pos}) + P(\text{Neg}) + \epsilon} \in [-1, +1]$$
- **Target-Based Directional Impact (Metode 2)**:
  $$\text{entity\_dir} = (\text{flag\_us} \times \text{polarity}) - (2.0 \times \text{flag\_idr} \times \text{polarity})$$
  Sentimen positif pada AS mendorong USD naik (+); sentimen positif pada Indonesia mendorong Rupiah menguat (-).
- **Interaksi GPR Pelemahan**: Saat densitas geopolitik (`gpr_dens`) tinggi dan sentimen negatif (`polarity < 0`), tekanan *safe-haven* mendorong USD naik / Rupiah melemah (`gpr_weak`).""")

add_code(r"""US_CASE  = re.compile(r"\b(U\.S\.?|USA|US|USD|Fed|FOMC|Powell|Yellen|Bessent|Treasury|Wall Street|White House|Trump|Biden)\b")
US_LOWER = re.compile(r"\b(united states|america|american|federal reserve|the fed|dollar|greenback)\b")
IDR_CASE = re.compile(r"\b(IDR|BI|IHSG|Rupiah|Indonesia|Indonesian|Jakarta|Jokowi|Prabowo|Sri Mulyani|Perry Warjiyo|Bank Indonesia)\b")
IDR_LOWER = re.compile(r"\b(indonesia|indonesian|rupiah|jakarta|bank indonesia|bi rate|bi-rate|ihsg)\b")

GPR_SET = {
    "war", "military", "conflict", "geopolitic", "geopolitical", "threat", "sanction", "sanctions",
    "tension", "tensions", "crisis", "attack", "invasion", "nuclear", "terrorism", "terrorist",
    "missile", "hostility", "clash", "rebel", "insurgency", "troops", "sovereignty", "escalation",
    "retaliation", "dispute", "standoff", "blockade", "weapon", "drills", "border", "combat",
    "tariff", "tariffs", "perang", "militer", "konflik", "geopolitik", "ancaman", "sanksi", "krisis"
}

daily_art_records = []
for idx, row in df_raw.iterrows():
    raw_titles = str(row[title_col]).split(" | ") if pd.notna(row[title_col]) else []
    raw_bodies = str(row[body_col]).split(" | ") if pd.notna(row[body_col]) else []
    
    art_feats = []
    for i, t in enumerate(raw_titles):
        t_c = clean_text_fn(t)
        if not t_c or t_c.lower() == "nan":
            continue
        b_c = clean_text_fn(raw_bodies[i]) if i < len(raw_bodies) else ""
        if b_c.lower() == "nan":
            b_c = ""
            
        scores = fin_map.get(t_c, {"finbert_pos": 0.5, "finbert_neg": 0.5})
        pos, neg = scores["finbert_pos"], scores["finbert_neg"]
        
        # Polaritas strictly positif vs negatif (eliminasi dilusi netral)
        pol_bin    = 1.0 if pos >= neg else -1.0
        pol_margin = (pos - neg) / (pos + neg + 1e-8)
        
        full_t = f"{t_c} {b_c}"
        us_title  = int(bool(US_CASE.search(t_c) or US_LOWER.search(t_c.lower())))
        idr_title = int(bool(IDR_CASE.search(t_c) or IDR_LOWER.search(t_c.lower())))
        us_cnt  = len(US_CASE.findall(full_t)) + len(US_LOWER.findall(full_t.lower()))
        idr_cnt = len(IDR_CASE.findall(full_t)) + len(IDR_LOWER.findall(full_t.lower()))
        
        w_us  = 1.0 if (us_title or us_cnt >= 2) else 0.0
        w_idr = 1.0 if (idr_title or idr_cnt >= 1) else 0.0
        
        toks  = full_t.lower().split()
        gpr_d = sum(1 for w in toks if w in GPR_SET) / max(len(toks), 1)
        
        art_feats.append({
            "pol_bin": pol_bin,
            "pol_margin": pol_margin,
            "flag_us": w_us,
            "flag_idr": w_idr,
            "entity_dir_bin": (w_us * pol_bin) - (w_idr * 2.0 * pol_bin),
            "entity_dir_margin": (w_us * pol_margin) - (w_idr * 2.0 * pol_margin),
            "gpr_dens": gpr_d,
            "gpr_weak_bin": gpr_d * (-pol_bin),
            "gpr_weak_margin": gpr_d * (-pol_margin),
        })
        
    if art_feats:
        agg = pd.DataFrame(art_feats).mean().to_dict()
        agg["has_news"] = 1.0
    else:
        # Hari tanpa berita: pastikan nilai netral/nol (bukan dipaksa +1)
        agg = {k: 0.0 for k in ["pol_bin", "pol_margin", "flag_us", "flag_idr",
                                "entity_dir_bin", "entity_dir_margin", "gpr_dens",
                                "gpr_weak_bin", "gpr_weak_margin"]}
        agg["has_news"] = 0.0
        
    agg[date_col] = row[date_col]
    agg["rate_clean"] = row["rate_clean"]
    for k in range(5):
        agg[f"topic_prob_{k}"] = row[f"topic_prob_{k}"]
    daily_art_records.append(agg)

text_daily_base = pd.DataFrame(daily_art_records).set_index(date_col).sort_index()
print("Ringkasan Fitur Entitas & Polaritas Harian (Sample 5 baris pertama):")
display(text_daily_base[["has_news", "flag_us", "flag_idr", "pol_bin", "pol_margin", "entity_dir_margin", "gpr_weak_margin"]].head())""")

# ----------------- CELL 13 & 14: DUAL-ENCODER & SELECTIVE PCA -----------------
add_md(r"""## 6. Ekstraksi Dual-Encoder Embedding & Selective PCA
Mengekstrak representasi semantik laten dari judul (`bert-base-uncased`) dan teks lengkap (`allenai/longformer-base-4096`), lalu mereduksi dimensinya menjadi **6 komponen PCA** (di-*fit* murni pada Train set).""")

add_code(r"""cache_dir = Path("data/processed")
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

daily_emb = pd.DataFrame(X_concat_all).groupby(df_raw[date_col]).mean().sort_index()
train_dates_emb = daily_emb.index <= train_end
pca = PCA(n_components=6, random_state=SEED)
pca.fit(daily_emb[train_dates_emb])
pca_features = pd.DataFrame(pca.transform(daily_emb), index=daily_emb.index,
                            columns=[f"pca_{i}" for i in range(6)])
print(f"PCA Variance Explained (6 komponen utama): {pca.explained_variance_ratio_.sum():.4f}")""")

# ----------------- CELL 15 & 16: TIME-SERIES FEATURES & LOG RETURN -----------------
add_md(r"""## 7. Penyelarasan Zona Waktu Teks & Fitur Deret Waktu Ringkas (`log_return`)
- **Target Prediksi**: $\text{log\_ret}_t = \ln\left(\frac{\text{rate}_t}{\text{rate}_{t-1}}\right)$, dan target biner arah $\text{target\_direction}_t = \mathbb{I}(\text{log\_ret}_t > 0)$.
- **Penyelarasan Teks US $\to$ WIB (`_lag1`)**: Karena berita CNBC mengikuti jam perdagangan AS (siang AS $\approx$ malam WIB setelah fiksasi kurs BI), fitur teks dari $t-1$ diselaraskan secara ketat tanpa kebocoran untuk memprediksi pergerakan hari $t$.
- **Fitur Time-Series Ringkas ($t-1$ Zero Lookahead)**:
  1. `ret_lag1`, `ret_lag2`: Return logaritmik 1 dan 2 hari lalu.
  2. `vol_5d`: Volatilitas rolling 5 hari.
  3. `consec_up_days`: Counter jumlah hari naik/turun berturut-turut (*reversal exhaustion*).
  4. `dist_ma20`: Jarak deviasi kurs kemarin terhadap Moving Average 20 hari (`rate / MA20 - 1`).""")

add_code(r"""text_daily_df = text_daily_base.join(pca_features)

# 1. Target Log Return & Arah
text_daily_df["log_ret"]          = np.log(text_daily_df["rate_clean"] / text_daily_df["rate_clean"].shift(1))
text_daily_df["target_direction"] = (text_daily_df["log_ret"] > 0).astype(int) # 1 = USD Naik / IDR Melemah

# 2. Penyelarasan Zona Waktu (Lagged Text Features t-1)
base_txt_cols = [
    "has_news", "pol_bin", "pol_margin", "flag_us", "flag_idr",
    "entity_dir_bin", "entity_dir_margin", "gpr_dens", "gpr_weak_bin", "gpr_weak_margin"
]
for c in base_txt_cols:
    text_daily_df[f"{c}_lag1"] = text_daily_df[c].shift(1)

# 3. Fitur Deret Waktu Ringkas (Strictly Lagged t-1)
text_daily_df["ret_lag1"] = text_daily_df["log_ret"].shift(1)
text_daily_df["ret_lag2"] = text_daily_df["log_ret"].shift(2)
text_daily_df["vol_5d"]   = text_daily_df["log_ret"].shift(1).rolling(5).std()

# Indikator Reversal 1: consecutive_up_days (dihitung hingga t-1)
consecs, cur = [], 0
for r in text_daily_df["log_ret"]:
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
text_daily_df["consec_up_days"] = pd.Series(consecs, index=text_daily_df.index).shift(1)

# Indikator Reversal 2: Jarak Deviasi MA 20 Hari (rate / MA20 - 1 hingga t-1)
ma20 = text_daily_df["rate_clean"].shift(1).rolling(20).mean()
text_daily_df["dist_ma20"] = text_daily_df["rate_clean"].shift(1) / ma20 - 1.0

daily_valid = text_daily_df.dropna(subset=["log_ret", "ret_lag2", "vol_5d", "dist_ma20", "consec_up_days", "pol_bin_lag1"]).copy()
print(f"Total baris harian valid bebas kebocoran: {len(daily_valid)}")""")

# ----------------- CELL 17 & 18: STAGE 1 DIRECTION CLASSIFIER -----------------
add_md(r"""## 8. Pemodelan Tahap 1: Klasifikasi Arah Teks (USD vs IDR) dengan 5-Fold OOF
Menggunakan fitur teks entitas US/IDR, polaritas bipolar, dan GPR pelemahan yang diselaraskan ke zona waktu WIB (`_lag1`) untuk mengklasifikasikan apakah USD atau IDR yang akan menguat. Prediksi probabilitas data latih dihasilkan via **5-Fold Out-of-Fold (OOF)** guna menghindari *in-sample stacking bias*.""")

add_code(r"""tr_m = daily_valid.index <= train_end
va_m = (daily_valid.index > train_end) & (daily_valid.index <= val_end)
te_m = daily_valid.index > val_end

text_clf_cols = [f"{c}_lag1" for c in base_txt_cols]

scaler_txt_s1 = StandardScaler().fit(daily_valid.loc[tr_m, text_clf_cols])
X_txt_tr = scaler_txt_s1.transform(daily_valid.loc[tr_m, text_clf_cols])
X_txt_va = scaler_txt_s1.transform(daily_valid.loc[va_m, text_clf_cols])
X_txt_te = scaler_txt_s1.transform(daily_valid.loc[te_m, text_clf_cols])

y_dir_tr = daily_valid.loc[tr_m, "target_direction"].values
y_dir_va = daily_valid.loc[va_m, "target_direction"].values
y_dir_te = daily_valid.loc[te_m, "target_direction"].values

# 5-Fold Out-of-Fold (OOF) dengan GradientBoostingClassifier
kf = KFold(n_splits=5, shuffle=False)
oof_pred_prob = np.zeros(len(X_txt_tr))

for fold_tr, fold_val in kf.split(X_txt_tr):
    clf_fold = GradientBoostingClassifier(n_estimators=40, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=SEED)
    clf_fold.fit(X_txt_tr[fold_tr], y_dir_tr[fold_tr])
    oof_pred_prob[fold_val] = clf_fold.predict_proba(X_txt_tr[fold_val])[:, 1]

clf_full = GradientBoostingClassifier(n_estimators=40, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=SEED).fit(X_txt_tr, y_dir_tr)

daily_valid.loc[tr_m, "pred_text_prob_up"] = oof_pred_prob
daily_valid.loc[va_m, "pred_text_prob_up"] = clf_full.predict_proba(X_txt_va)[:, 1]
daily_valid.loc[te_m, "pred_text_prob_up"] = clf_full.predict_proba(X_txt_te)[:, 1]

p_te_s1 = daily_valid.loc[te_m, "pred_text_prob_up"].values
pred_te_s1 = (p_te_s1 >= 0.5).astype(int)
acc_s1 = accuracy_score(y_dir_te, pred_te_s1) * 100
bacc_s1 = balanced_accuracy_score(y_dir_te, pred_te_s1) * 100
cm_s1 = confusion_matrix(y_dir_te, pred_te_s1).ravel()

print(f"Stage 1 Klasifikasi Arah Teks (Test Set): Accuracy={acc_s1:.2f}%, Balanced Acc={bacc_s1:.2f}%, CM(TN,FP,FN,TP)={cm_s1}")""")

# ----------------- CELL 19 & 20: FEATURE GROUPS -----------------
add_md(r"""## 9. Penggabungan Master Dataset & Definisi Grup Fitur
Menggabungkan 5 fitur deret waktu ringkas dengan sinyal arah teks Stage 1 (`pred_text_prob_up`, `entity_dir_margin_lag1`, `gpr_weak_margin_lag1`).""")

add_code(r"""GROUPS = {
    "ts_basic": ["ret_lag1", "ret_lag2", "vol_5d"],
    "ts_reversal": ["consec_up_days", "dist_ma20"],
    "text_direction": ["pred_text_prob_up", "entity_dir_margin_lag1", "gpr_weak_margin_lag1"],
    "topik": [f"topic_prob_{i}" for i in range(5)],
    "selective_pca": [f"pca_{i}" for i in range(6)]
}

core_feature_cols = GROUPS["ts_basic"] + GROUPS["ts_reversal"] + GROUPS["text_direction"]
print(f"Fitur Inti Stage 2 ({len(core_feature_cols)} fitur): {core_feature_cols}")""")

# ----------------- CELL 21 & 22: STAGE 2 DIRECTION-AWARE MLP & BENCHMARK -----------------
add_md(r"""## 10. Pemodelan Tahap 2: Direction-Aware MLP & Ridge Benchmark (`log_return`)
Melatih model hilir untuk memprediksi `log_return`. Selain **Ridge Regressor**, kami melatih **Direction-Aware PyTorch MLP** menggunakan fungsi rugi komposit:
$$\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{dir}} \cdot \text{Softplus}\left(-3 \cdot \text{sign}(y) \cdot \hat{y}\right)$$
Fungsi rugi ini menjaga kalibrasi besaran *error* (RMSE) sekaligus memberi penalti jika arah prediksi melenceng, sehingga model mampu memprediksi pergerakan **naik maupun turun secara seimbang** (*Balanced Accuracy* tinggi).""")

add_code(r"""def compute_metrics(y_true, y_pred, y_train_ref):
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    
    denom = np.sum((y_true - y_train_ref) ** 2)
    r2 = float(1 - np.sum(err ** 2) / denom) if denom > 0 else 0.0
    
    valid_sign = (y_true != 0) & (y_pred != 0)
    da = float(np.mean(np.sign(y_true[valid_sign]) == np.sign(y_pred[valid_sign])) * 100) if valid_sign.any() else 50.0
    bacc = float(balanced_accuracy_score((y_true > 0).astype(int), (y_pred > 0).astype(int)) * 100)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "DirAcc (%)": da, "BalAcc (%)": bacc, "R2_oos": r2}

scaler_master = StandardScaler().fit(daily_valid.loc[tr_m, core_feature_cols])
X_tr = scaler_master.transform(daily_valid.loc[tr_m, core_feature_cols])
X_va = scaler_master.transform(daily_valid.loc[va_m, core_feature_cols])
X_te = scaler_master.transform(daily_valid.loc[te_m, core_feature_cols])

y_tr = daily_valid.loc[tr_m, "log_ret"].values
y_va = daily_valid.loc[va_m, "log_ret"].values
y_te = daily_valid.loc[te_m, "log_ret"].values

mu_y, std_y = y_tr.mean(), y_tr.std()
y_tr_scaled = (y_tr - mu_y) / std_y
y_va_scaled = (y_va - mu_y) / std_y

# 1. Master Ridge Regressor
ridge_model = Ridge(alpha=100.0, random_state=SEED).fit(X_tr, y_tr_scaled)
pred_ridge_te = ridge_model.predict(X_te) * std_y + mu_y

# 2. Direction-Aware PyTorch MLP
class DirAwareMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_dir_aware_mlp(Xtr, ytr_s, ytr_raw, Xva, yva_s, yva_raw, Xte, dir_weight=0.25, epochs=200):
    set_seed(SEED)
    model = DirAwareMLP(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(ytr_s, dtype=torch.float32, device=DEVICE)
    st = torch.tensor(np.where(ytr_raw > 0, 1.0, -1.0), dtype=torch.float32, device=DEVICE)
    
    Xv = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(yva_s, dtype=torch.float32, device=DEVICE)
    sv = torch.tensor(np.where(yva_raw > 0, 1.0, -1.0), dtype=torch.float32, device=DEVICE)
    
    mu_ratio = float(mu_y / std_y)
    best_val, best_w = float("inf"), None
    tr_hist, va_hist = [], []
    
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        ps = model(Xt)
        mse_tr = torch.mean((ps - yt) ** 2)
        dir_tr = torch.mean(torch.nn.functional.softplus(-st * (ps + mu_ratio) * 3.0))
        loss = mse_tr + dir_weight * dir_tr
        loss.backward()
        opt.step()
        
        model.eval()
        with torch.no_grad():
            pv = model(Xv)
            mse_va = torch.mean((pv - yv) ** 2).item()
            dir_va = torch.mean(torch.nn.functional.softplus(-sv * (pv + mu_ratio) * 3.0)).item()
            vloss = mse_va + dir_weight * dir_va
            
        tr_hist.append(loss.item())
        va_hist.append(vloss)
        if vloss < best_val:
            best_val = vloss
            best_w = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_w)
    model.eval()
    with torch.no_grad():
        p_te = model(torch.tensor(Xte, dtype=torch.float32, device=DEVICE)).cpu().numpy() * std_y + mu_y
    return p_te, tr_hist, va_hist

pred_mlp_mse_te, _, _ = train_dir_aware_mlp(X_tr, y_tr_scaled, y_tr, X_va, y_va_scaled, y_va, X_te, dir_weight=0.0)
pred_mlp_dir_te, train_loss_hist, val_loss_hist = train_dir_aware_mlp(X_tr, y_tr_scaled, y_tr, X_va, y_va_scaled, y_va, X_te, dir_weight=0.25)

# 3. Ensemble Blend (50% Dir-Aware MLP + 50% Ridge)
pred_blend_te = 0.5 * pred_mlp_dir_te + 0.5 * pred_ridge_te

results_list = [
    {"Model": "Baseline: Tebak Return 0",           **compute_metrics(y_te, np.zeros_like(y_te), mu_y)},
    {"Model": "Baseline: Tebak Rata-rata Train",    **compute_metrics(y_te, np.full_like(y_te, mu_y), mu_y)},
    {"Model": "Basic MLP (Pure MSE)",               **compute_metrics(y_te, pred_mlp_mse_te, mu_y)},
    {"Model": "Master Ridge Regressor (Shrinkage)", **compute_metrics(y_te, pred_ridge_te, mu_y)},
    {"Model": "Direction-Aware MLP (dir_w=0.25)",   **compute_metrics(y_te, pred_mlp_dir_te, mu_y)},
    {"Model": "Ensemble Blend (DirMLP + Ridge)",    **compute_metrics(y_te, pred_blend_te, mu_y)}
]

eval_df = pd.DataFrame(results_list).set_index("Model")
print("\n=================================== HASIL EVALUASI PREDIKSI LOG_RETURN (TEST SET) ===================================")
display(eval_df.round(6))""")

# ----------------- CELL 23 & 24: ABLATION STUDY -----------------
add_md(r"""## 11. Studi Ablasi Bertahap (Bebas Kebocoran)
Menguji kontribusi nyata setiap kelompok fitur terhadap kualitas prediksi **Direction-Aware MLP**:
1. Deret waktu dasar (`ret_lag1`, `ret_lag2`, `vol_5d`).
2. + Indikator Reversal & Momentum (`consec_up_days`, `dist_ma20`).
3. + Sinyal Arah Teks Entitas & GPR Stage 1 (`pred_text_prob_up`, `entity_dir_margin_lag1`, `gpr_weak_margin_lag1`).""")

add_code(r"""ablation_steps = [
    ("1. TS Dasar (Lag1, Lag2, Vol5d)", ["ts_basic"]),
    ("2. + Reversal (ConsecUp, DistMA20)", ["ts_basic", "ts_reversal"]),
    ("3. + Sinyal Arah Teks (Entity & GPR)", ["ts_basic", "ts_reversal", "text_direction"])
]

ablation_rows = []
for label, group_keys in ablation_steps:
    selected_cols = sum((GROUPS[k] for k in group_keys), [])
    sc = StandardScaler().fit(daily_valid.loc[tr_m, selected_cols])
    X_tr_a = sc.transform(daily_valid.loc[tr_m, selected_cols])
    X_va_a = sc.transform(daily_valid.loc[va_m, selected_cols])
    X_te_a = sc.transform(daily_valid.loc[te_m, selected_cols])
    
    p_a, _, _ = train_dir_aware_mlp(X_tr_a, y_tr_scaled, y_tr, X_va_a, y_va_scaled, y_va, X_te_a, dir_weight=0.25)
    ablation_rows.append({"Tahapan Fitur": label, **compute_metrics(y_te, p_a, mu_y)})

ablation_df = pd.DataFrame(ablation_rows).set_index("Tahapan Fitur")
print("\n=================================== HASIL STUDI ABLASI (TEST SET) ===================================")
display(ablation_df.round(6))""")

# ----------------- CELL 25 & 26: 5-PANEL VISUALIZATIONS -----------------
add_md(r"""## 12. Visualisasi Komprehensif (5 Panel Analisis)
1. Bar Chart **RMSE & MAE**.
2. Bar Chart **Directional Accuracy & Balanced Accuracy (%)**.
3. Time Series **Log Return Aktual vs Prediksi Direction-Aware MLP**.
4. Kurva **Learning Loss per Epoch (Train vs Val)**.
5. Koefisien Pengaruh Fitur (**Feature Importance** Ridge Regressor).""")

add_code(r"""fig = plt.figure(figsize=(20, 10), dpi=150)
gs = fig.add_gridspec(2, 6, hspace=0.35, wspace=0.35)

eval_plot = eval_df.drop(["Baseline: Tebak Return 0", "Baseline: Tebak Rata-rata Train"])
x_idx = np.arange(len(eval_plot))
w = 0.35

# 1. RMSE & MAE Bar Chart
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.bar(x_idx - w/2, eval_plot["RMSE"], w, label="RMSE", color="#2a9d8f")
ax1.bar(x_idx + w/2, eval_plot["MAE"], w, label="MAE", color="#e76f51")
ax1.axhline(0.002902, color="red", linestyle="--", alpha=0.7, label="Baseline Mean (~0.00290)")
ax1.set_title("Perbandingan Error (RMSE & MAE) Test Set", fontweight="bold", fontsize=11)
ax1.set_xticks(x_idx)
ax1.set_xticklabels(eval_plot.index, rotation=20, ha="right", fontsize=8.5)
ax1.set_ylabel("Nilai Error")
ax1.legend(fontsize=8.5)
ax1.grid(axis="y", linestyle=":", alpha=0.6)

# 2. Directional Accuracy & Balanced Accuracy Bar Chart
ax2 = fig.add_subplot(gs[0, 2:4])
ax2.bar(x_idx - w/2, eval_plot["DirAcc (%)"], w, label="DirAcc (%)", color="#457b9d")
ax2.bar(x_idx + w/2, eval_plot["BalAcc (%)"], w, label="Balanced Acc (%)", color="#f4a261")
ax2.axhline(50.0, color="gray", linestyle="--", label="Tebak Koin / Baseline BalAcc (50%)")
ax2.set_title("Akurasi Arah: DirAcc vs Balanced Acc (%)", fontweight="bold", fontsize=11)
ax2.set_xticks(x_idx)
ax2.set_xticklabels(eval_plot.index, rotation=20, ha="right", fontsize=8.5)
ax2.set_ylabel("Akurasi (%)")
ax2.set_ylim(45, 65)
ax2.legend(fontsize=8.5)
ax2.grid(axis="y", linestyle=":", alpha=0.6)

# 3. Time Series Plot: Log Return
ax3 = fig.add_subplot(gs[0, 4:6])
test_dates = daily_valid.loc[te_m].index
ax3.plot(test_dates, y_te, label="Log Return Aktual", color="black", alpha=0.65, linewidth=1.1)
ax3.plot(test_dates, pred_mlp_dir_te, label="Prediksi Dir-Aware MLP", color="#e63946", linestyle="--", linewidth=1.3)
ax3.axhline(0, color="gray", linewidth=0.7)
ax3.set_title("Deret Waktu Log Return: Aktual vs Prediksi", fontweight="bold", fontsize=11)
ax3.set_ylabel("Log Return")
ax3.legend(fontsize=8.5)
ax3.grid(True, linestyle=":", alpha=0.6)

# 4. Kurva Konvergensi MLP Loss per Epoch
ax4 = fig.add_subplot(gs[1, 0:3])
ax4.plot(train_loss_hist, label="Training Composite Loss", color="#1d3557", linewidth=1.5)
ax4.plot(val_loss_hist, label="Validation Composite Loss", color="#e63946", linestyle="--", linewidth=1.5)
ax4.set_title("Kurva Pembelajaran Direction-Aware MLP (Loss vs Epoch)", fontweight="bold", fontsize=11)
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Composite Loss (MSE + Sign Penalty)")
ax4.legend(fontsize=9)
ax4.grid(True, linestyle=":", alpha=0.6)

# 5. Feature Importance (Koefisien Ridge)
ax5 = fig.add_subplot(gs[1, 3:6])
coef_series = pd.Series(ridge_model.coef_, index=core_feature_cols).sort_values()
colors = ["#e76f51" if c < 0 else "#2a9d8f" for c in coef_series.values]
ax5.barh(coef_series.index, coef_series.values, color=colors)
ax5.axvline(0, color="black", linewidth=0.8)
ax5.set_title("Pengaruh Fitur Terhadap Log Return (Ridge Coefficients)", fontweight="bold", fontsize=11)
ax5.set_xlabel("Nilai Koefisien Regresi Terstandar")
ax5.grid(axis="x", linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()""")

# ----------------- CELL 27 & 28: SAVE REPORTS -----------------
add_md(r"""## 13. Simpan Hasil Evaluasi & Model Artifacts
Menyimpan ringkasan metrik evaluasi dan dataset master ke direktori `reports/`.""")

add_code(r"""reports_dir = Path("percobaan task 3/reports")
reports_dir.mkdir(parents=True, exist_ok=True)

eval_csv_path = reports_dir / "evaluasi_directional_logreturn.csv"
ablation_csv_path = reports_dir / "ablation_directional_logreturn.csv"
master_csv_path = reports_dir / "master_dataset_directional.csv"

eval_df.to_csv(eval_csv_path)
ablation_df.to_csv(ablation_csv_path)
daily_valid.to_csv(master_csv_path)

print("Laporan hasil eksperimen bebas kebocoran berhasil disimpan:")
print(f"  1. Ringkasan Evaluasi Multi-Model: {eval_csv_path}")
print(f"  2. Hasil Studi Ablasi           : {ablation_csv_path}")
print(f"  3. Master Dataset Bersih        : {master_csv_path}")""")

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

with open("percobaan task 3/trial3_pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb_dict, f, indent=1, ensure_ascii=False)
print("Berhasil menulis ke percobaan task 3/trial3_pipeline.ipynb")

with open("scratch/trial3_pipeline_clean.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb_dict, f, indent=1, ensure_ascii=False)
print("Berhasil menulis ke scratch/trial3_pipeline_clean.ipynb")
