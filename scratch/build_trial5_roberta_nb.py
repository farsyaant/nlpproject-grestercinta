import json

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

# ----------------- CELL 0: HEADER -----------------
add_md(r"""# Pipeline Eksperimen Task 3 (Trial 5): Prediksi Arah Kurs Berbasis RoBERTa Proporsional (Freeze Weights, Tanpa Cache, + Hyperparameter Tuning)
### Right-Sized Architecture: Frozen Twitter-RoBERTa SPSM, Frozen RoBERTa MCM + PCA, LDA, & Hyperparameter Tuning

Notebook ini menjalankan versi **proporsional, cepat, dan bebas *overfitting*** untuk dataset ~800–1.000 hari:
1. **Tanpa Cache Disk**: Seluruh penyaringan teks, skor FinBERT/LLM, serta ekstraksi embedding **dihitung segar langsung di memori** pada setiap eksekusi.
2. **Freeze Weights (`requires_grad = False`) untuk Semua Encoder Pretrained**:
   - Seluruh bobot **FinBERT**, **Twitter-RoBERTa-Base (SPSM)**, dan **RoBERTa-Base (MCM)** dibekukan agar ekstraksi fitur 768-d berjalan **sangat cepat** dalam satu lintasan `torch.no_grad()`, lalu hanya melatih *head* ringan di atas embedding beku tersebut.
3. **Hyperparameter Tuning Bebas Kebocoran**:
   - Tuning **Topik LDA & Logistic Regression** (`n_components` & `C`), **SPSM MLP Regression Head** (`hidden`, `dropout`, `lr`, `wd`), **MCM MLP Classification Head** (`hidden`, `dropout`, `lr`, `wd`), **PCA + LogReg** (`n_components`, `C`), **Hybrid Meta-Classifier** (`TimeSeriesSplit` pada Validation Set), dan **Ridge Regression** (`alpha`).""")

# ----------------- CELL 1 & 2: INSTALL -----------------
add_md(r"""## 0. Instalasi Library Pendukung""")

add_code(r"""!pip install -q transformers torch scikit-learn openai""")

# ----------------- CELL 3 & 4: IMPORTS & CONFIG -----------------
add_md(r"""## 1. Konfigurasi Lingkungan & Backbone Proporsional (Freeze Weights, No Cache)""")

add_code(r"""import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import re
import json
import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, log_loss,
    roc_auc_score, roc_curve, confusion_matrix, mean_squared_error, mean_absolute_error
)

warnings.filterwarnings("ignore")

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_OPENAI_API = False
OPENAI_MODEL   = "gpt-4o"

SPSM_BACKBONE = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MCM_BACKBONE  = "roberta-base"

def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()
print(f"Device aktif  : {DEVICE}")
print(f"Backbone SPSM : {SPSM_BACKBONE} (Freeze Weights + Tuned MLP Regression Head)")
print(f"Backbone MCM  : {MCM_BACKBONE} (Freeze Weights + Tuned Lean Head & PCA)")
print(f"Mode LLM      : {'OpenAI ' + OPENAI_MODEL if USE_OPENAI_API else 'Local Teacher Fallback (Frozen FinBERT, No Cache)'}")""")

# ----------------- CELL 5 & 6: LOAD DATA & BINARY LABELING -----------------
add_md(r"""## 2. Pemuatan Dataset & Pembuatan Label Arah Hari Berikutnya ($d \to d+1$)
* **Label `0`**: $\text{rate}_{d+1} < \text{rate}_d$ (Kurs besok turun / Rupiah menguat).
* **Label `1`**: $\text{rate}_{d+1} \ge \text{rate}_d$ (Kurs besok sama atau naik / USD menguat).""")

add_code(r"""DATA_CANDIDATES = [
    Path("data/processed/merged_data_clean.csv"),
    Path("../data/processed/merged_data_clean.csv"),
    Path("/kaggle/input/datasets/graceee12/process-dataset/processed/merged_data_clean.csv"),
    Path("merged_data_clean.csv")
]

DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), None)
if DATA_PATH is None:
    raise FileNotFoundError("File merged_data_clean.csv tidak ditemukan!")

df_raw = pd.read_csv(DATA_PATH)
date_col = "Tanggal_dt" if "Tanggal_dt" in df_raw.columns else ("tanggal" if "tanggal" in df_raw.columns else "Tanggal")
df_raw[date_col] = pd.to_datetime(df_raw[date_col])
df_raw = df_raw.sort_values(date_col).reset_index(drop=True)

if "Kurs Jual" in df_raw.columns and "Kurs Beli" in df_raw.columns:
    df_raw["rate_close"] = (df_raw["Kurs Jual"] + df_raw["Kurs Beli"]) / 2.0
elif "rate_clean" in df_raw.columns:
    df_raw["rate_close"] = df_raw["rate_clean"]
else:
    df_raw["rate_close"] = df_raw["rate"]

df_raw["rate_next"] = df_raw["rate_close"].shift(-1)
df_raw["log_ret_next"] = np.log(df_raw["rate_next"] / df_raw["rate_close"])
df_raw["label_dir_next"] = (df_raw["rate_next"] >= df_raw["rate_close"]).astype(int)

df_base = df_raw.dropna(subset=["rate_next"]).reset_index(drop=True)
title_col = "all_title" if "all_title" in df_base.columns else "title"
body_col  = "all_full_text" if "all_full_text" in df_base.columns else ("all_fulltext" if "all_fulltext" in df_base.columns else "full_text")

print(f"Total hari bursa berlabel d->d+1: {len(df_base)} hari")
print(f"Distribusi Label Awal (0=Turun, 1=Sama/Naik): {(df_base['label_dir_next']==0).sum()} vs {(df_base['label_dir_next']==1).sum()}")""")

# ----------------- CELL 7 & 8: 2-PASS FILTER & SENTIMENT SCORING (IN-MEMORY, NO CACHE) -----------------
add_md(r"""## 3. Penyaringan 2-Tahap & Pelabelan Sentimen Kontinu (Di Memori, Tanpa Cache)
Menghasilkan `df_fx` (**Level Harian**) dan `df_art` (**Level Per Artikel**) secara langsung di memori menggunakan `ProsusAI/finbert` dengan **Freeze Weights** (`requires_grad = False`).""")

add_code(r"""FX_MACRO_PATTERN = re.compile(
    r"\b(dollar|usd|rupiah|idr|fed|federal reserve|powell|yellen|bessent|interest rate|rate hike|rate cut|"
    r"inflation|cpi|treasury|yield|bond|tariff|tariffs|sanction|sanctions|oil|crude|opec|currency|fx|"
    r"forex|exchange rate|central bank|bank indonesia|export|import|trade balance|recession|gdp|economy)\b",
    re.IGNORECASE
)

def llm_two_pass_and_score_openai(title, body):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    p1 = client.chat.completions.create(
        model=OPENAI_MODEL, temperature=0, max_tokens=5,
        messages=[{"role": "user", "content": f"Is this article relevant to USD/IDR macro/FX? Reply YES or NO.\nTitle: {title}\nBody: {body[:1200]}"}]
    ).choices[0].message.content.strip().upper()
    if "YES" not in p1:
        return False, "", 0.0, "Irrelevant"
    p2 = client.chat.completions.create(
        model=OPENAI_MODEL, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "user", "content": f"Extract FX/macro paragraphs, score sentiment polarity [-1.0, +1.0], and give 1-sentence explanation as JSON keys 'extracted_text', 'sentiment_score', 'explanation'.\nTitle: {title}\nBody: {body[:3000]}"}]
    ).choices[0].message.content
    d = json.loads(p2)
    return True, str(d.get("extracted_text", title)), float(np.clip(d.get("sentiment_score", 0.0), -1.0, 1.0)), str(d.get("explanation", ""))

print("Menjalankan 2-Pass FX Filtering & Inferensi Frozen FinBERT langsung di memori (Tanpa Cache)...")
can_use_openai = USE_OPENAI_API and bool(os.getenv("OPENAI_API_KEY"))
fin_map = {}

if not can_use_openai:
    all_t = []
    for s in df_base[title_col].dropna():
        for p in str(s).split(" | "):
            pc = p.strip()
            if pc and pc.lower() != "nan":
                all_t.append(pc)
    uniq_t = sorted(list(set(all_t)), key=len)
    ftok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    fmod = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(DEVICE)
    for param in fmod.parameters():
        param.requires_grad = False
    fmod.eval()
    with torch.no_grad():
        for i in range(0, len(uniq_t), 64):
            b = uniq_t[i:i+64]
            inp = ftok(b, padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                probs = torch.softmax(fmod(**inp).logits, dim=-1).cpu().numpy()
            for ts, pa in zip(b, probs):
                fin_map[ts] = {"finbert_pos": float(pa[0]), "finbert_neg": float(pa[1]), "finbert_neu": float(pa[2])}
    del fmod, ftok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

daily_records, art_records = [], []
for idx, row in df_base.iterrows():
    raw_t = str(row[title_col]).split(" | ") if pd.notna(row[title_col]) else []
    raw_b = str(row[body_col]).split(" | ") if pd.notna(row[body_col]) else []
    kept_segs, scores, expls = [], [], []
    for j, t in enumerate(raw_t):
        tc = t.strip()
        if not tc or tc.lower() == "nan":
            continue
        bc = raw_b[j].strip() if j < len(raw_b) else ""
        if bc.lower() == "nan":
            bc = ""
        if can_use_openai:
            ok, ext, pol, exp = llm_two_pass_and_score_openai(tc, bc)
            if not ok or not ext.strip():
                continue
        else:
            if not FX_MACRO_PATTERN.search(f"{tc} {bc}"):
                continue
            sents = re.split(r"(?<=[.!?])\s+|\n+", bc)
            rel_paras = [s.strip() for s in sents if len(s.strip()) > 25 and FX_MACRO_PATTERN.search(s)]
            ext = f"{tc}. " + " ".join(rel_paras[:4]) if rel_paras else tc
            sc = fin_map.get(tc, {"finbert_pos": 0.33, "finbert_neg": 0.33})
            pol = float(np.clip(sc["finbert_pos"] - sc["finbert_neg"], -1.0, 1.0))
            kw = FX_MACRO_PATTERN.findall(ext)
            exp = f"Polaritas {pol:+.2f} [{', '.join(sorted(set(k.lower() for k in kw))[:3])}]"

        kept_segs.append(ext)
        scores.append(pol)
        expls.append(exp)
        art_records.append({
            date_col: row[date_col],
            "article_text": ext,
            "llm_sentiment": pol,
            "llm_explanation": exp
        })

    if kept_segs:
        daily_records.append({
            date_col: row[date_col],
            "rate_close": row["rate_close"],
            "log_ret_next": row["log_ret_next"],
            "label_dir_next": row["label_dir_next"],
            "filtered_fx_text": " ".join(kept_segs),
            "llm_sentiment": float(np.mean(scores)),
            "llm_explanation": " | ".join(expls[:2]),
            "kept_articles": len(kept_segs)
        })

df_fx  = pd.DataFrame(daily_records)
df_art = pd.DataFrame(art_records)

# Pembagian Kronologis (70% Train, 15% Val, 15% Test)
n_fx = len(df_fx)
n_tr, n_va = int(n_fx * 0.70), int(n_fx * 0.15)
idx_tr = np.arange(0, n_tr)
idx_va = np.arange(n_tr, n_tr + n_va)
idx_te = np.arange(n_tr + n_va, n_fx)

train_end_date = df_fx[date_col].iloc[idx_tr[-1]]
val_end_date   = df_fx[date_col].iloc[idx_va[-1]]

print(f"Level Harian -> Train={len(idx_tr)} | Val={len(idx_va)} | Test={len(idx_te)}")
print(f"Level Artikel (untuk SPSM) -> Train={(df_art[date_col] <= train_end_date).sum()} | Val={((df_art[date_col] > train_end_date) & (df_art[date_col] <= val_end_date)).sum()} | Test={(df_art[date_col] > val_end_date).sum()}")""")

# ----------------- CELL 9 & 10: LDA TOPIC MODELING + TUNING -----------------
add_md(r"""## 4. Ekstraksi & Hyperparameter Tuning Topik LDA + Sentimen Pembanding
Melakukan tuning jumlah topik `LDA (n_components in [5, 8, 12])` dan regularisasi `LogisticRegression C` di Validation Set.""")

add_code(r"""cv = CountVectorizer(max_features=1500, stop_words="english", min_df=3, max_df=0.85)
X_bow_tr = cv.fit_transform(df_fx.iloc[idx_tr]["filtered_fx_text"])
X_bow_va = cv.transform(df_fx.iloc[idx_va]["filtered_fx_text"])
X_bow_te = cv.transform(df_fx.iloc[idx_te]["filtered_fx_text"])

y_tr_dir = df_fx.iloc[idx_tr]["label_dir_next"].values
y_va_dir = df_fx.iloc[idx_va]["label_dir_next"].values
y_te_dir = df_fx.iloc[idx_te]["label_dir_next"].values

# Tuning LDA n_components & LogisticRegression C
best_lda_score, best_lda_cfg = -1e9, None
best_lda_bundle = None
for n_top in [5, 8, 12]:
    lda_c = LatentDirichletAllocation(n_components=n_top, learning_method="online", max_iter=15, random_state=SEED, n_jobs=1)
    X_l_tr = lda_c.fit_transform(X_bow_tr)
    X_l_va = lda_c.transform(X_bow_va)
    X_l_te = lda_c.transform(X_bow_te)
    for c_val in [0.05, 0.1, 0.5, 1.0, 2.0]:
        clf_c = LogisticRegression(C=c_val, class_weight="balanced", random_state=SEED).fit(X_l_tr, y_tr_dir)
        p_va = clf_c.predict_proba(X_l_va)[:, 1]
        bal = balanced_accuracy_score(y_va_dir, (p_va >= 0.5).astype(int))
        ll = log_loss(y_va_dir, np.clip(p_va, 1e-6, 1 - 1e-6))
        sc = bal - 0.05 * ll
        if sc > best_lda_score:
            best_lda_score = sc
            best_lda_cfg = {"n_topics": n_top, "C": c_val}
            best_lda_bundle = (lda_c, X_l_tr, X_l_va, X_l_te, clf_c)

lda, X_lda_tr, X_lda_va, X_lda_te, clf_lda = best_lda_bundle
prob_lda_va = clf_lda.predict_proba(X_lda_va)[:, 1]
prob_lda_te = clf_lda.predict_proba(X_lda_te)[:, 1]
print(f"Best LDA Params: {best_lda_cfg}")

vocab = cv.get_feature_names_out()
print(f"Kata Kunci {best_lda_cfg['n_topics']} Topik LDA:")
for tid, topic in enumerate(lda.components_):
    print(f"  Topik {tid}: {', '.join([vocab[i] for i in topic.argsort()[:-8:-1]])}")

# Tuning Classifier Sentimen LLM
best_sent_sc, best_sent_c, clf_sent = -1e9, 0.5, None
for c_val in [0.05, 0.1, 0.5, 1.0, 2.0]:
    cand = LogisticRegression(C=c_val, class_weight="balanced", random_state=SEED).fit(df_fx.iloc[idx_tr][["llm_sentiment"]], y_tr_dir)
    p_va = cand.predict_proba(df_fx.iloc[idx_va][["llm_sentiment"]])[:, 1]
    sc = balanced_accuracy_score(y_va_dir, (p_va >= 0.5).astype(int)) - 0.05 * log_loss(y_va_dir, np.clip(p_va, 1e-6, 1 - 1e-6))
    if sc > best_sent_sc:
        best_sent_sc, best_sent_c, clf_sent = sc, c_val, cand
prob_sent_te = clf_sent.predict_proba(df_fx.iloc[idx_te][["llm_sentiment"]])[:, 1]
print(f"Best Sentiment LogReg C={best_sent_c}")""")

# ----------------- CELL 11 & 12: FROZEN SPSM ENCODER + TUNED REGRESSION HEAD -----------------
add_md(r"""## 5. Modul 1: **SPSM** (*Sentiment Prediction Sub-Module*) — Frozen `Twitter-RoBERTa` + Tuned Regression Head
Seluruh bobot encoder **`Twitter-RoBERTa-Base`** (`cardiffnlp/twitter-roberta-base-sentiment-latest`) **dibekukan (*Freeze Weights*)** sehingga ekstraksi representasi artikel 768-d selesai dalam satu lintasan `torch.no_grad()` yang sangat cepat (tanpa *fine-tuning* encoder yang lambat dan tanpa cache disk), kemudian dilakukan **Hyperparameter Tuning** pada kepala regresi (`hidden`, `dropout`, `lr`, `wd`) dengan *Early Stopping*.""")

add_code(r"""# 1. Ekstraksi 768-d Per Artikel dengan Frozen Twitter-RoBERTa (Cepat, Tanpa Cache)
try:
    tok_spsm = AutoTokenizer.from_pretrained(SPSM_BACKBONE)
    enc_spsm = AutoModel.from_pretrained(SPSM_BACKBONE).to(DEVICE)
    spsm_name_used = SPSM_BACKBONE
except Exception as e:
    print(f"Fallback SPSM ke roberta-base karena: {e}")
    spsm_name_used = "roberta-base"
    tok_spsm = AutoTokenizer.from_pretrained(spsm_name_used)
    enc_spsm = AutoModel.from_pretrained(spsm_name_used).to(DEVICE)

for param in enc_spsm.parameters():
    param.requires_grad = False
enc_spsm.eval()

art_texts_all = df_art["article_text"].tolist()
print(f"Mengekstrak Frozen {spsm_name_used} 768-d untuk {len(art_texts_all)} artikel...")
art_emb_chunks = []
with torch.no_grad():
    for i in range(0, len(art_texts_all), 64):
        inp = tok_spsm(art_texts_all[i:i+64], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            cls_v = enc_spsm(**inp).last_hidden_state[:, 0, :]
        art_emb_chunks.append(cls_v.cpu().numpy().astype(np.float32))

E_art = np.vstack(art_emb_chunks)
del enc_spsm, tok_spsm
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

tr_a = (df_art[date_col] <= train_end_date).values
va_a = ((df_art[date_col] > train_end_date) & (df_art[date_col] <= val_end_date)).values

sc_spsm = StandardScaler().fit(E_art[tr_a])
Ea_tr = sc_spsm.transform(E_art[tr_a])
Ea_va = sc_spsm.transform(E_art[va_a])
Ea_all = sc_spsm.transform(E_art)
ya_tr = df_art.loc[tr_a, "llm_sentiment"].values.astype(np.float32)
ya_va = df_art.loc[va_a, "llm_sentiment"].values.astype(np.float32)

class TunedSPSMHead(nn.Module):
    def __init__(self, in_dim=768, hidden=64, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1), nn.Tanh())
    def forward(self, x):
        return self.head(x).squeeze(-1)

# 2. Hyperparameter Tuning Kepala Regresi SPSM di Validation Set Artikel
Xat = torch.tensor(Ea_tr, dtype=torch.float32, device=DEVICE)
yat = torch.tensor(ya_tr, dtype=torch.float32, device=DEVICE)
Xav = torch.tensor(Ea_va, dtype=torch.float32, device=DEVICE)
yav = torch.tensor(ya_va, dtype=torch.float32, device=DEVICE)
Xa_all_t = torch.tensor(Ea_all, dtype=torch.float32, device=DEVICE)

best_sv, best_spsm_cfg, best_spsm_model, spsm_tr_hist, spsm_va_hist = float("inf"), None, None, [], []
crit_s = nn.MSELoss()

for h in [32, 64]:
    for dr in [0.25, 0.4]:
        for lr in [1e-3, 3e-3]:
            for wd in [1e-3, 1e-2]:
                set_seed(SEED)
                mod_c = TunedSPSMHead(Ea_tr.shape[1], hidden=h, dropout=dr).to(DEVICE)
                opt_c = torch.optim.AdamW(mod_c.parameters(), lr=lr, weight_decay=wd)
                bv, bw, wait, th, vh = float("inf"), None, 0, [], []
                for ep in range(120):
                    mod_c.train(); opt_c.zero_grad()
                    loss = crit_s(mod_c(Xat), yat)
                    loss.backward(); opt_c.step()
                    mod_c.eval()
                    with torch.no_grad():
                        vl = crit_s(mod_c(Xav), yav).item()
                    th.append(loss.item()); vh.append(vl)
                    if vl < bv - 1e-5:
                        bv, bw, wait = vl, copy.deepcopy(mod_c.state_dict()), 0
                    else:
                        wait += 1
                        if wait >= 15:
                            break
                if bv < best_sv:
                    mod_c.load_state_dict(bw)
                    best_sv = bv
                    best_spsm_cfg = {"hidden": h, "dropout": dr, "lr": lr, "wd": wd}
                    best_spsm_model = mod_c
                    spsm_tr_hist, spsm_va_hist = th, vh

print(f"Best SPSM Frozen Head Params (Article Val MSE={best_sv:.5f}): {best_spsm_cfg}")
best_spsm_model.eval()
with torch.no_grad():
    df_art["spsm_pred"] = best_spsm_model(Xa_all_t).cpu().numpy()

daily_mean_spsm = df_art.groupby(date_col)["spsm_pred"].mean()
spsm_daily_preds = df_fx[date_col].map(daily_mean_spsm).fillna(0.0).values.astype(np.float32)

spsm_pred_tr, spsm_pred_va, spsm_pred_te = spsm_daily_preds[idx_tr], spsm_daily_preds[idx_va], spsm_daily_preds[idx_te]
spsm_mse = mean_squared_error(df_fx.iloc[idx_te]["llm_sentiment"], spsm_pred_te)
spsm_rmse = float(np.sqrt(spsm_mse))
spsm_mae = mean_absolute_error(df_fx.iloc[idx_te]["llm_sentiment"], spsm_pred_te)
print(f"Evaluasi SPSM Harian (Test Set) -> RMSE={spsm_rmse:.5f} | MSE={spsm_mse:.5f} | MAE={spsm_mae:.5f}")""")

# ----------------- CELL 13 & 14: FROZEN ROBERTA + TUNED MCM HEAD & TUNED PCA LOGREG -----------------
add_md(r"""## 6. Modul 2: **MCM** (*Market Classification Module*) — Frozen `RoBERTa-Base` + Tuned Head & Tuned PCA
Seluruh bobot `roberta-base` **dibekukan (*Freeze Weights*)** tanpa cache disk untuk mengekstrak representasi harian 768-d (*attention-masked mean pooling*), lalu dilakukan **Hyperparameter Tuning**:
1. **Tuned MCM Classifier Head**: Tuning `hidden` $\in [32, 64, 128]$, `dropout` $\in [0.35, 0.5]$, `lr` $\in [5\text{e-}4, 1\text{e-}3, 3\text{e-}3]$, `wd` $\in [0.01, 0.05]$ dengan *Early Stopping*.
2. **Tuned Frozen RoBERTa + PCA + LogReg**: Tuning `n_components` $\in [16, 32, 48]$ dan `C` $\in [0.01, 0.05, 0.1, 0.5, 1.0]$ di Validation Set.""")

add_code(r"""print(f"Mengekstrak embedding Frozen {MCM_BACKBONE} (768-d, No Cache) untuk {len(df_fx)} hari...")
tok_mcm = AutoTokenizer.from_pretrained(MCM_BACKBONE)
enc_mcm = AutoModel.from_pretrained(MCM_BACKBONE).to(DEVICE)
for param in enc_mcm.parameters():
    param.requires_grad = False
enc_mcm.eval()

texts_fx = df_fx["filtered_fx_text"].tolist()
embs_list = []
with torch.no_grad():
    for i in range(0, len(texts_fx), 16):
        inp = tok_mcm(texts_fx[i:i+16], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            out = enc_mcm(**inp).last_hidden_state
        mask = inp["attention_mask"].unsqueeze(-1).expand(out.size()).float()
        mean_v = torch.sum(out * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        embs_list.append(mean_v.cpu().numpy().astype(np.float32))
X_rob_daily = np.vstack(embs_list)
del enc_mcm, tok_mcm
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

sc_rob = StandardScaler().fit(X_rob_daily[idx_tr])
Xtr_rob = sc_rob.transform(X_rob_daily[idx_tr])
Xva_rob = sc_rob.transform(X_rob_daily[idx_va])
Xte_rob = sc_rob.transform(X_rob_daily[idx_te])

# 1. Tuning Lean MCM Head (768 -> hidden -> 2)
class LeanMCMHead(nn.Module):
    def __init__(self, in_dim=768, hidden=64, dropout=0.45):
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(hidden, 2)
        )
    def forward(self, x):
        return self.cls_head(x)

cnt_0, cnt_1 = max((y_tr_dir == 0).sum(), 1), max((y_tr_dir == 1).sum(), 1)
cw = torch.tensor([len(y_tr_dir)/(2.0*cnt_0), len(y_tr_dir)/(2.0*cnt_1)], dtype=torch.float32, device=DEVICE)
crit_m = nn.CrossEntropyLoss(weight=cw)

Xt_m = torch.tensor(Xtr_rob, dtype=torch.float32, device=DEVICE)
Xv_m = torch.tensor(Xva_rob, dtype=torch.float32, device=DEVICE)
Xte_m = torch.tensor(Xte_rob, dtype=torch.float32, device=DEVICE)
yt_m = torch.tensor(y_tr_dir, dtype=torch.long, device=DEVICE)
yv_m = torch.tensor(y_va_dir, dtype=torch.long, device=DEVICE)

best_mv, best_mcm_cfg, best_mcm_model, mcm_tr_hist, mcm_va_hist = float("inf"), None, None, [], []
for h in [32, 64]:
    for dr in [0.35, 0.5]:
        for lr in [1e-3, 3e-3]:
            for wd in [0.01, 0.05]:
                set_seed(SEED)
                m_cand = LeanMCMHead(768, hidden=h, dropout=dr).to(DEVICE)
                opt_m = torch.optim.AdamW(m_cand.parameters(), lr=lr, weight_decay=wd)
                bv, bw, wait, th, vh = float("inf"), None, 0, [], []
                for ep in range(100):
                    m_cand.train(); opt_m.zero_grad()
                    loss = crit_m(m_cand(Xt_m), yt_m)
                    loss.backward(); opt_m.step()
                    m_cand.eval()
                    with torch.no_grad():
                        vl = crit_m(m_cand(Xv_m), yv_m).item()
                    th.append(loss.item()); vh.append(vl)
                    if vl < bv - 1e-4:
                        bv, bw, wait = vl, copy.deepcopy(m_cand.state_dict()), 0
                    else:
                        wait += 1
                        if wait >= 15:
                            break
                if bv < best_mv:
                    m_cand.load_state_dict(bw)
                    best_mv = bv
                    best_mcm_cfg = {"hidden": h, "dropout": dr, "lr": lr, "wd": wd}
                    best_mcm_model = m_cand
                    mcm_tr_hist, mcm_va_hist = th, vh

print(f"Best Lean MCM Head Params (Val CE={best_mv:.4f}): {best_mcm_cfg}")
best_mcm_model.eval()
with torch.no_grad():
    mcm_prob_va = torch.softmax(best_mcm_model(Xv_m), dim=-1)[:, 1].cpu().numpy()
    mcm_prob_te = torch.softmax(best_mcm_model(Xte_m), dim=-1)[:, 1].cpu().numpy()

# 2. Tuning Frozen RoBERTa + PCA + LogisticRegression
best_pca_sc, best_pca_cfg, best_pca_bundle = -1e9, None, None
for n_comp in [16, 32, 48]:
    pca_c = PCA(n_components=n_comp, random_state=SEED).fit(Xtr_rob)
    Ztr, Zva, Zte = pca_c.transform(Xtr_rob), pca_c.transform(Xva_rob), pca_c.transform(Xte_rob)
    for c_val in [0.01, 0.05, 0.1, 0.5, 1.0]:
        clf_c = LogisticRegression(C=c_val, class_weight="balanced", random_state=SEED).fit(Ztr, y_tr_dir)
        p_va = clf_c.predict_proba(Zva)[:, 1]
        bal = balanced_accuracy_score(y_va_dir, (p_va >= 0.5).astype(int))
        auc = roc_auc_score(y_va_dir, p_va) if len(np.unique(y_va_dir)) > 1 else 0.5
        sc = bal + 0.25 * auc
        if sc > best_pca_sc:
            best_pca_sc = sc
            best_pca_cfg = {"n_components": n_comp, "C": c_val}
            best_pca_bundle = (pca_c, clf_c, clf_c.predict_proba(Zva)[:, 1], clf_c.predict_proba(Zte)[:, 1])

pca_opt, clf_pca_opt, pca_prob_va, pca_prob_te = best_pca_bundle
print(f"Best Frozen RoBERTa PCA+LogReg Params: {best_pca_cfg}")""")

# ----------------- CELL 15 & 16: HYBRID FUSION & TUNED EVALUATION -----------------
add_md(r"""## 7. Hybrid Fusion (`MCM + PCA + SPSM + LDA`) & Evaluasi Menyeluruh (Tuned)""")

add_code(r"""# Meta-Classifier menggabungkan sinyal MCM, PCA, Sentimen SPSM, dan Topik LDA (Tuning C dengan TimeSeriesSplit pada Validation Set)
X_meta_va = np.hstack([mcm_prob_va[:, None], pca_prob_va[:, None], spsm_pred_va[:, None], X_lda_va])
X_meta_te = np.hstack([mcm_prob_te[:, None], pca_prob_te[:, None], spsm_pred_te[:, None], X_lda_te])

sc_meta = StandardScaler().fit(X_meta_va)
Z_meta_va = sc_meta.transform(X_meta_va)
Z_meta_te = sc_meta.transform(X_meta_te)

tscv_m = TimeSeriesSplit(n_splits=4)
best_meta_c, best_meta_sc = 0.1, -1e9
for c_val in [0.01, 0.05, 0.1, 0.5, 1.0]:
    f_scores = []
    for ftr, fva in tscv_m.split(Z_meta_va):
        if len(np.unique(y_va_dir[ftr])) < 2 or len(np.unique(y_va_dir[fva])) < 2:
            continue
        cm = LogisticRegression(C=c_val, class_weight="balanced", random_state=SEED).fit(Z_meta_va[ftr], y_va_dir[ftr])
        f_scores.append(roc_auc_score(y_va_dir[fva], cm.predict_proba(Z_meta_va[fva])[:, 1]))
    mean_f = float(np.mean(f_scores)) if f_scores else 0.5
    if mean_f > best_meta_sc:
        best_meta_sc, best_meta_c = mean_f, c_val

meta_clf = LogisticRegression(C=best_meta_c, class_weight="balanced", random_state=SEED).fit(Z_meta_va, y_va_dir)
prob_hybrid_te = meta_clf.predict_proba(Z_meta_te)[:, 1]
print(f"Best Hybrid Meta-Classifier C={best_meta_c} (TS-CV AUC={best_meta_sc:.4f})")

def eval_cls_metrics(name, y_true, prob):
    pred = (prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, pred) * 100
    bal = balanced_accuracy_score(y_true, pred) * 100
    f1_m = f1_score(y_true, pred, average="macro") * 100
    f1_w = f1_score(y_true, pred, average="weighted") * 100
    auc = roc_auc_score(y_true, prob)
    mae_p = float(mean_absolute_error(y_true, prob))
    mse_p = float(mean_squared_error(y_true, prob))
    rmse_p = float(np.sqrt(mse_p))
    return {
        "Model": name, "Accuracy (%)": acc, "Balanced Acc (%)": bal,
        "Macro F1 (%)": f1_m, "Weighted F1 (%)": f1_w, "ROC-AUC": auc,
        "MAE (Prob)": mae_p, "MSE (Brier)": mse_p, "RMSE (Prob)": rmse_p
    }

eval_rows = [
    eval_cls_metrics("Baseline: Mayoritas (Selalu 1)", y_te_dir, np.ones_like(mcm_prob_te)),
    eval_cls_metrics(f"Pembanding 1: Topik LDA-{best_lda_cfg['n_topics']} + LogReg", y_te_dir, prob_lda_te),
    eval_cls_metrics("Pembanding 2: Sentimen LLM + LogReg", y_te_dir, prob_sent_te),
    eval_cls_metrics("Modul MCM (Frozen RoBERTa Tuned)", y_te_dir, mcm_prob_te),
    eval_cls_metrics(f"Frozen RoBERTa PCA-{best_pca_cfg['n_components']} + LogReg", y_te_dir, pca_prob_te),
    eval_cls_metrics("Hybrid Fusion Tuned (MCM+PCA+SPSM+LDA)", y_te_dir, prob_hybrid_te),
]

eval_cls_df = pd.DataFrame(eval_rows).set_index("Model")
print("\n===================== HASIL KLASIFIKASI ARAH & GALAT PROBABILITAS (TEST SET) =====================")
display(eval_cls_df.round(4))

# Evaluasi Regresi Besaran Log Return Besok (d -> d+1) + Tuning Alpha Ridge
y_tr_ret = df_fx.iloc[idx_tr]["log_ret_next"].values
y_va_ret = df_fx.iloc[idx_va]["log_ret_next"].values
y_te_ret = df_fx.iloc[idx_te]["log_ret_next"].values
mu_ret, std_ret = y_tr_ret.mean(), y_tr_ret.std()
y_tr_ret_s = (y_tr_ret - mu_ret) / std_ret
y_va_ret_s = (y_va_ret - mu_ret) / std_ret

def tune_ridge(Xtr_r, Xva_r, Xte_r):
    best_a, best_vm = 50.0, float("inf")
    best_m = None
    for a in [5.0, 20.0, 50.0, 100.0, 300.0, 500.0, 1000.0]:
        m = Ridge(alpha=a, random_state=SEED).fit(Xtr_r, y_tr_ret_s)
        vm = mean_squared_error(y_va_ret, m.predict(Xva_r) * std_ret + mu_ret)
        if vm < best_vm:
            best_vm, best_a, best_m = vm, a, m
    return best_m.predict(Xte_r) * std_ret + mu_ret, best_a

def eval_reg_metrics(name, y_true, y_pred):
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    denom = np.sum((y_true - mu_ret) ** 2)
    r2 = float(1 - np.sum(err ** 2) / denom) if denom > 0 else 0.0
    v = (y_true != 0) & (y_pred != 0)
    da = float(np.mean(np.sign(y_true[v]) == np.sign(y_pred[v])) * 100) if v.any() else 50.0
    ba = float(balanced_accuracy_score((y_true >= 0).astype(int), (y_pred >= 0).astype(int)) * 100)
    return {"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "DirAcc (%)": da, "BalAcc (%)": ba, "R2_oos": r2}

pred_reg_lda, a_lda = tune_ridge(X_lda_tr, X_lda_va, X_lda_te)
pred_reg_sent, a_sent = tune_ridge(df_fx.iloc[idx_tr][["llm_sentiment"]], df_fx.iloc[idx_va][["llm_sentiment"]], df_fx.iloc[idx_te][["llm_sentiment"]])
pred_reg_pca, a_pca = tune_ridge(pca_opt.transform(Xtr_rob), pca_opt.transform(Xva_rob), pca_opt.transform(Xte_rob))

# Tuning alpha Ridge Hybrid via TimeSeriesSplit di Validation Set
best_hyb_a, best_hyb_cv = 50.0, float("inf")
for a in [5.0, 20.0, 50.0, 100.0, 300.0, 500.0]:
    f_errs = []
    for ftr, fva in tscv_m.split(Z_meta_va):
        mh = Ridge(alpha=a, random_state=SEED).fit(Z_meta_va[ftr], y_va_ret_s[ftr])
        f_errs.append(mean_squared_error(y_va_ret[fva], mh.predict(Z_meta_va[fva]) * std_ret + mu_ret))
    m_e = float(np.mean(f_errs))
    if m_e < best_hyb_cv:
        best_hyb_cv, best_hyb_a = m_e, a
reg_hyb = Ridge(alpha=best_hyb_a, random_state=SEED).fit(Z_meta_va, y_va_ret_s)
pred_reg_hyb = reg_hyb.predict(Z_meta_te) * std_ret + mu_ret

eval_reg_df = pd.DataFrame([
    eval_reg_metrics("Baseline: Tebak Return 0", y_te_ret, np.zeros_like(y_te_ret)),
    eval_reg_metrics("Baseline: Rata-rata Train", y_te_ret, np.full_like(y_te_ret, mu_ret)),
    eval_reg_metrics(f"Regresi Topik LDA (a={a_lda:.0f})", y_te_ret, pred_reg_lda),
    eval_reg_metrics(f"Regresi Sentimen LLM (a={a_sent:.0f})", y_te_ret, pred_reg_sent),
    eval_reg_metrics(f"Regresi Frozen RoBERTa PCA (a={a_pca:.0f})", y_te_ret, pred_reg_pca),
    eval_reg_metrics(f"Regresi Hybrid Tuned (a={best_hyb_a:.0f})", y_te_ret, pred_reg_hyb),
]).set_index("Model")
print("\n===================== HASIL REGRESI LOG RETURN BESOK (TUNED RIDGE) =====================")
display(eval_reg_df.round(6))""")

# ----------------- CELL 17 & 18: VISUALIZATION & SAVE -----------------
add_md(r"""## 8. Visualisasi Komprehensif (4 Panel Evaluasi & Kurva Early Stopping)""")

add_code(r"""fig, axes = plt.subplots(2, 2, figsize=(18, 11), dpi=140)

# Panel 1: Accuracy, Balanced Accuracy, & Macro F1
ax1 = axes[0, 0]
x_idx = np.arange(len(eval_cls_df))
w = 0.25
ax1.bar(x_idx - w, eval_cls_df["Accuracy (%)"], w, label="Accuracy (%)", color="#2a9d8f")
ax1.bar(x_idx, eval_cls_df["Balanced Acc (%)"], w, label="Balanced Acc (%)", color="#e9c46a")
ax1.bar(x_idx + w, eval_cls_df["Macro F1 (%)"], w, label="Macro F1 (%)", color="#e76f51")
ax1.axhline(50.0, color="gray", linestyle="--", label="Tebak Koin (50%)")
ax1.set_title("Perbandingan Metrik Klasifikasi Arah Besok (Tuned, Test Set)", fontweight="bold")
ax1.set_xticks(x_idx)
ax1.set_xticklabels(eval_cls_df.index, rotation=22, ha="right", fontsize=8.5)
ax1.set_ylabel("Persentase (%)")
ax1.set_ylim(25, 72)
ax1.legend(fontsize=8.5)
ax1.grid(axis="y", linestyle=":", alpha=0.6)

# Panel 2: ROC Curve
ax2 = axes[0, 1]
for lbl, pr, col in [
    (f"Topik LDA-{best_lda_cfg['n_topics']}", prob_lda_te, "#264653"),
    ("MCM Tuned (Frozen RoBERTa)", mcm_prob_te, "#f4a261"),
    (f"Frozen RoBERTa PCA-{best_pca_cfg['n_components']}", pca_prob_te, "#2a9d8f"),
    ("Hybrid Fusion Tuned", prob_hybrid_te, "#e63946")
]:
    fpr, tpr, _ = roc_curve(y_te_dir, pr)
    ax2.plot(fpr, tpr, label=f"{lbl} (AUC={roc_auc_score(y_te_dir, pr):.3f})", color=col, linewidth=1.8)
ax2.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax2.set_title("Kurva ROC Prediksi Arah Hari Berikutnya", fontweight="bold")
ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
ax2.legend(fontsize=8.5); ax2.grid(True, linestyle=":", alpha=0.6)

# Panel 3: Confusion Matrix Hybrid Fusion
ax3 = axes[1, 0]
cm_hyb = confusion_matrix(y_te_dir, (prob_hybrid_te >= 0.5).astype(int))
sns.heatmap(cm_hyb, annot=True, fmt="d", cmap="Blues", ax=ax3,
            xticklabels=["Pred 0 (Turun)", "Pred 1 (Naik/Sama)"],
            yticklabels=["Aktual 0 (Turun)", "Aktual 1 (Naik/Sama)"])
ax3.set_title("Confusion Matrix: Hybrid Fusion Tuned (Test Set)", fontweight="bold")

# Panel 4: Early Stopping Learning Curve (Lean MCM Head Tuned)
ax4 = axes[1, 1]
ax4.plot(mcm_tr_hist, label="MCM Train Cross-Entropy", color="#1d3557", linewidth=1.5)
ax4.plot(mcm_va_hist, label="MCM Val Cross-Entropy", color="#e63946", linestyle="--", linewidth=1.5)
ax4.set_title(f"Kurva Pembelajaran MCM Head Tuned (Berhenti di Epoch {len(mcm_va_hist)})", fontweight="bold")
ax4.set_xlabel("Epoch"); ax4.set_ylabel("Weighted Cross-Entropy Loss")
ax4.legend(fontsize=9); ax4.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()

reports_dir = Path("percobaan task 3/reports")
reports_dir.mkdir(parents=True, exist_ok=True)
out_csv = reports_dir / "evaluasi_trial5_roberta_spsm_mcm.csv"
out_reg_csv = reports_dir / "evaluasi_trial5_regresi_logret.csv"
eval_cls_df.to_csv(out_csv)
eval_reg_df.to_csv(out_reg_csv)
print(f"Ringkasan hasil Trial 5 disimpan ke:\n  - Klasifikasi & Brier: {out_csv}\n  - Regresi Log Return : {out_reg_csv}")""")

nb_dict = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python", "version": "3.11"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = "percobaan task 3/trial5_roberta_spsm_mcm_lda.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb_dict, f, indent=1, ensure_ascii=False)
print(f"Notebook Trial 5 (Freeze Weights + No Cache + Tuning) berhasil diperbarui: {out_path} ({len(cells)} sel)")
