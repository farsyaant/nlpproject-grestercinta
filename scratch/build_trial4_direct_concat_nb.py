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
add_md(r"""# Eksperimen Direct Embedding Concatenation: BERT (Title) + Longformer (Full-Text)
### Pipeline Langsung (Freeze Weights, Tanpa Cache Disk, + Hyperparameter Tuning) $\to$ Model MLP & Machine Learning

Notebook ini menguji pendekatan **Direct Representation Learning** murni dari teks berita tanpa ekstraksi topik (BERTopic), leksikon GPR, atau skor sentimen terpisah:
1. **Judul Berita (`title`)** di-embed menggunakan **BERT** umum (`bert-base-uncased`, 768 dimensi).
2. **Isi Berita (`full_text`)** di-embed menggunakan **Longformer** (`allenai/longformer-base-4096`, 768 dimensi) guna menangkap konteks artikel panjang hingga ribuan token.
3. **Freeze Weights & Tanpa Cache**: Seluruh bobot encoder BERT & Longformer **dibekukan (`requires_grad = False`)** agar ekstraksi embedding sangat cepat (`eval()` + `torch.no_grad()`) dan dijalankan **langsung di memori tanpa membaca/menyimpan cache disk**.
4. **Concatenation ($1536$ dimensi)**: Kedua vektor digabungkan per artikel lalu diagregasi (*mean pooling*) ke level harian.
5. **Hyperparameter Tuning Berbasis Validation Set**:
   - **Jalur A (Langsung 1536-d)**: Tuning hyperparameter **Direct Deep MLP** (`hidden`, `dropout`, `lr`, `wd`) dan **Ridge Regression** (`alpha`).
   - **Jalur B (1536-d + PCA Optimal)**: Tuning dimensi `PCA` (`[16, 32, 48]`), **Compact MLP**, **Ridge**, **Random Forest**, **Gradient Boosting (GBDT)**, dan bobot **Ensemble Blend**.
6. **Early Stopping Eksplisit**: Dilengkapi `patience` agar pelatihan MLP berhenti otomatis pada titik *Validation Loss* minimum.
7. **Evaluasi Lengkap**: Mengukur **RMSE, MAE, MSE, Directional Accuracy (%), Balanced Accuracy (%), dan $R^2_{\text{oos}}$** pada target `log_return`.""")

# ----------------- CELL 1 & 2: INSTALL -----------------
add_md(r"""## 0. Instalasi Library Pendukung""")

add_code(r"""!pip install -q transformers torch scikit-learn""")

# ----------------- CELL 3 & 4: IMPORTS & CONFIG -----------------
add_md(r"""## 1. Konfigurasi Environment, Seed, dan Model Pretrained""")

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

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, balanced_accuracy_score

warnings.filterwarnings("ignore")

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TITLE_MODEL    = "bert-base-uncased"
FULLTEXT_MODEL = "allenai/longformer-base-4096"

def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()
print(f"Device aktif          : {DEVICE}")
print(f"Model Title (BERT)    : {TITLE_MODEL} (Freeze Weights, No Cache)")
print(f"Model Fulltext (Long) : {FULLTEXT_MODEL} (Freeze Weights, No Cache)")""")

# ----------------- CELL 5 & 6: LOAD DATA -----------------
add_md(r"""## 2. Pemuatan Dataset, Target `log_return`, & Pembagian Kronologis (70/15/15)""")

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
    df_raw["rate_clean"] = (df_raw["Kurs Jual"] + df_raw["Kurs Beli"]) / 2.0
elif "rate_clean" not in df_raw.columns and "rate" in df_raw.columns:
    df_raw["rate_clean"] = df_raw["rate"]

df_raw["log_ret"] = np.log(df_raw["rate_clean"] / df_raw["rate_clean"].shift(1))

title_col = "all_title" if "all_title" in df_raw.columns else "title"
body_col  = "all_full_text" if "all_full_text" in df_raw.columns else ("all_fulltext" if "all_fulltext" in df_raw.columns else "full_text")

df_valid = df_raw.dropna(subset=["log_ret"]).reset_index(drop=True)

n_total = len(df_valid)
n_train = int(n_total * 0.70)
n_val   = int(n_total * 0.15)

idx_train = np.arange(0, n_train)
idx_val   = np.arange(n_train, n_train + n_val)
idx_test  = np.arange(n_train + n_val, n_total)

print(f"Total hari valid  : {n_total} hari")
print(f"Train Set         : {len(idx_train)} hari ({df_valid[date_col].iloc[idx_train[0]].date()} s/d {df_valid[date_col].iloc[idx_train[-1]].date()})")
print(f"Validation Set    : {len(idx_val)} hari ({df_valid[date_col].iloc[idx_val[0]].date()} s/d {df_valid[date_col].iloc[idx_val[-1]].date()})")
print(f"Test Set          : {len(idx_test)} hari ({df_valid[date_col].iloc[idx_test[0]].date()} s/d {df_valid[date_col].iloc[idx_test[-1]].date()})")""")

# ----------------- CELL 7 & 8: DIRECT EXTRACTION & CONCAT (NO CACHE, FREEZE WEIGHTS) -----------------
add_md(r"""## 3. Ekstraksi Embedding (BERT Title + Longformer Full-Text) & Concatenation (1536-d)
Setiap baris harian yang berisi banyak berita (`' | '`) dipecah terlebih dahulu ke **level artikel individual** agar tidak terjadi pemotongan judul maupun isi berita pada artikel ke-2 dst., diekstrak cepat dengan **Freeze Weights (`requires_grad = False`) langsung di memori tanpa cache disk**, lalu di-*concat* menjadi **1536 dimensi** dan dirata-ratakan per hari perdagangan.""")

add_code(r"""def clean_basic(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    return re.sub(r"\s+", " ", text).strip()

print("Memecah artikel per hari untuk ekstraksi embedding bebas truncation (Freeze Weights, No Cache)...")
art_titles, art_bodies, art_day_idx = [], [], []

for day_i, row in df_valid.iterrows():
    raw_t = str(row[title_col]).split(" | ") if pd.notna(row[title_col]) else []
    raw_b = str(row[body_col]).split(" | ") if pd.notna(row[body_col]) else []
    for j, t in enumerate(raw_t):
        tc = clean_basic(t)
        if not tc or tc.lower() == "nan":
            continue
        bc = clean_basic(raw_b[j]) if j < len(raw_b) else ""
        if bc.lower() == "nan":
            bc = ""
        art_titles.append(tc)
        art_bodies.append(bc if bc else tc)
        art_day_idx.append(day_i)

print(f"Total artikel yang diekstrak: {len(art_titles)} artikel dari {len(df_valid)} hari bursa.")

def extract_mean_embeddings(texts, model_name, max_len, batch_size):
    print(f"  -> Memuat & menjalankan {model_name} [Freeze Weights] (max_len={max_len}, bs={batch_size})...")
    tok = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModel.from_pretrained(model_name).to(DEVICE)

    # Bekukan seluruh bobot encoder agar inferensi berjalan sangat cepat
    for param in mod.parameters():
        param.requires_grad = False
    mod.eval()

    chunks = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            b_texts = texts[i:i+batch_size]
            inp = tok(b_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEVICE)
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                out = mod(**inp).last_hidden_state
            mask = inp["attention_mask"].unsqueeze(-1).expand(out.size()).float()
            mean_vec = torch.sum(out * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            chunks.append(mean_vec.cpu().numpy().astype(np.float32))

    del mod, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.vstack(chunks)

# 1. Judul menggunakan BERT biasa (Freeze)
emb_title = extract_mean_embeddings(art_titles, TITLE_MODEL, max_len=64, batch_size=64)

# 2. Full-text menggunakan Longformer (Freeze; fallback ke BERT 512 hanya jika memori CPU/GPU tidak cukup)
try:
    emb_fulltext = extract_mean_embeddings(art_bodies, FULLTEXT_MODEL, max_len=1024, batch_size=16)
except Exception as e:
    print(f"Peringatan Longformer ({e}), beralih ke {TITLE_MODEL} max_len=512.")
    emb_fulltext = extract_mean_embeddings(art_bodies, TITLE_MODEL, max_len=512, batch_size=32)

# 3. Concat Title (768) + Fulltext (768) -> 1536 dimensi per artikel
emb_art_1536 = np.hstack([emb_title, emb_fulltext])

# 4. Agregasi (Mean Pooling) per hari perdagangan di memori
X_all = np.zeros((len(df_valid), 1536), dtype=np.float32)
day_series = pd.Series(art_day_idx)
for d_idx, grp in day_series.groupby(day_series):
    X_all[d_idx] = emb_art_1536[grp.index].mean(axis=0)

print(f"Dimensi matriks fitur X_all (in-memory, no cache): {X_all.shape}")""")

# ----------------- CELL 9 & 10: PREPARE SCALING & PCA TUNING -----------------
add_md(r"""## 4. Persiapan Data Pemodelan: Direct 1536-d & Tuning Reduksi PCA
Menyiapkan dua set fitur (semua *fit* murni pada Train Set agar bebas *leakage*):
1. `X_1536`: Vektor hasil concat langsung (`1536` fitur terstandarisasi).
2. `X_pca`: Reduksi dimensi **PCA** yang jumlah komponennya (`n_components` $\in [16, 24, 32, 48]$) dituning berdasarkan *Validation MSE*.""")

add_code(r"""y_all = df_valid["log_ret"].values.astype(np.float32)
y_train, y_val, y_test = y_all[idx_train], y_all[idx_val], y_all[idx_test]

# Standarisasi Target (z-score berdasarkan Train set)
mu_y, std_y = y_train.mean(), y_train.std()
y_train_s = (y_train - mu_y) / std_y
y_val_s   = (y_val - mu_y) / std_y

# Jalur A: Direct 1536-d
scaler_1536 = StandardScaler().fit(X_all[idx_train])
X_tr_1536 = scaler_1536.transform(X_all[idx_train])
X_va_1536 = scaler_1536.transform(X_all[idx_val])
X_te_1536 = scaler_1536.transform(X_all[idx_test])

# Jalur B: Tuning jumlah komponen PCA pada Validation Set
best_pca_k, best_pca_val = 32, float("inf")
best_pca_obj = None
for k_pca in [16, 24, 32, 48]:
    pca_cand = PCA(n_components=k_pca, random_state=SEED).fit(X_tr_1536)
    r_cand = Ridge(alpha=500.0, random_state=SEED).fit(pca_cand.transform(X_tr_1536), y_train_s)
    p_va_c = r_cand.predict(pca_cand.transform(X_va_1536)) * std_y + mu_y
    vmse = mean_squared_error(y_val, p_va_c)
    if vmse < best_pca_val:
        best_pca_val = vmse
        best_pca_k = k_pca
        best_pca_obj = pca_cand

X_tr_pca = best_pca_obj.transform(X_tr_1536)
X_va_pca = best_pca_obj.transform(X_va_1536)
X_te_pca = best_pca_obj.transform(X_te_1536)

print(f"PCA Optimal (Tuned): n_components={best_pca_k} | Variance Explained (Train): {best_pca_obj.explained_variance_ratio_.sum():.4f}")""")

# ----------------- CELL 11 & 12: MODELING MLP & ML WITH HYPERPARAMETER TUNING -----------------
add_md(r"""## 5. Hyperparameter Tuning & Pelatihan Model MLP (Early Stopping) + Machine Learning
Melakukan *grid search* pada **Validation Set** (kronologis, bebas *leakage*) untuk memilih hyperparameter optimal tiap model:
- **Jalur A (Direct 1536-d)**: Tuning `Direct Deep MLP` (`hidden`, `dropout`, `lr`, `wd`) & `Ridge` (`alpha`).
- **Jalur B (PCA-Tuned)**: Tuning `Compact MLP`, `Ridge`, `Random Forest` (`n_estimators`, `max_depth`, `min_samples_leaf`), `Gradient Boosting` (`n_estimators`, `max_depth`, `learning_rate`), serta bobot `Ensemble Blend`.""")

add_code(r"""def compute_metrics(y_true, y_pred, y_train_mean):
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))

    denom = np.sum((y_true - y_train_mean) ** 2)
    r2 = float(1 - np.sum(err ** 2) / denom) if denom > 0 else 0.0

    valid_sign = (y_true != 0) & (y_pred != 0)
    da = float(np.mean(np.sign(y_true[valid_sign]) == np.sign(y_pred[valid_sign])) * 100) if valid_sign.any() else 50.0
    ba = float(balanced_accuracy_score((y_true > 0).astype(int), (y_pred > 0).astype(int)) * 100)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "DirAcc (%)": da, "BalAcc (%)": ba, "R2_oos": r2}

class FlexibleMLP(nn.Module):
    def __init__(self, in_dim, hidden1=64, hidden2=16, dropout=0.45):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.Tanh(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_pytorch_mlp(Xtr, Xva, Xte, hidden1=64, hidden2=16, dropout=0.45, lr=1e-3, wd=0.05, max_epochs=150, patience=20):
    set_seed(SEED)
    model = FlexibleMLP(Xtr.shape[1], hidden1=hidden1, hidden2=hidden2, dropout=dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()

    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_train_s, dtype=torch.float32, device=DEVICE)
    Xv = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(y_val_s, dtype=torch.float32, device=DEVICE)

    best_val, best_w, wait = float("inf"), None, 0
    tr_hist, va_hist = [], []

    for ep in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = criterion(model(Xt), yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vloss = criterion(model(Xv), yv).item()

        tr_hist.append(loss.item())
        va_hist.append(vloss)

        if vloss < best_val - 1e-5:
            best_val = vloss
            best_w = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_w is not None:
        model.load_state_dict(best_w)
    model.eval()
    with torch.no_grad():
        pred_va = model(Xv).cpu().numpy() * std_y + mu_y
        pred_te = model(torch.tensor(Xte, dtype=torch.float32, device=DEVICE)).cpu().numpy() * std_y + mu_y
    return pred_te, pred_va, best_val, tr_hist, va_hist

def tune_mlp(Xtr, Xva, Xte, arch_list, dropouts, lrs, wds, patience=20):
    best_v, best_cfg, best_res = float("inf"), None, None
    for (h1, h2) in arch_list:
        for d in dropouts:
            for lr in lrs:
                for wd in wds:
                    pte, pva, bv, th, vh = train_pytorch_mlp(
                        Xtr, Xva, Xte, hidden1=h1, hidden2=h2, dropout=d, lr=lr, wd=wd, patience=patience
                    )
                    if bv < best_v:
                        best_v = bv
                        best_cfg = {"hidden1": h1, "hidden2": h2, "dropout": d, "lr": lr, "wd": wd}
                        best_res = (pte, pva, th, vh)
    return (*best_res, best_cfg, best_v)

# --- 1. TUNING JALUR A: DIRECT 1536-d ---
pred_mlp_direct, pred_mlp_direct_va, tr_hist_1536, va_hist_1536, cfg_mlp_1536, bv_mlp_1536 = tune_mlp(
    X_tr_1536, X_va_1536, X_te_1536,
    arch_list=[(64, 16), (128, 32)], dropouts=[0.4, 0.55], lrs=[5e-4, 1e-3], wds=[0.05], patience=18
)
print(f"[Direct 1536-d] Best MLP Params (Val Loss={bv_mlp_1536:.5f}): {cfg_mlp_1536}")

best_a_1536, bv_r1536, ridge_1536 = None, float("inf"), None
for a in [500.0, 2000.0, 10000.0, 20000.0, 50000.0]:
    cand = Ridge(alpha=a, random_state=SEED).fit(X_tr_1536, y_train_s)
    vmse = mean_squared_error(y_val, cand.predict(X_va_1536) * std_y + mu_y)
    if vmse < bv_r1536:
        bv_r1536, best_a_1536, ridge_1536 = vmse, a, cand
pred_ridge_direct = ridge_1536.predict(X_te_1536) * std_y + mu_y
print(f"[Direct 1536-d] Best Ridge alpha={best_a_1536} (Val MSE={bv_r1536:.8f})")

# --- 2. TUNING JALUR B: PCA-TUNED ---
pred_mlp_pca, pred_mlp_pca_va, tr_hist_pca, va_hist_pca, cfg_mlp_pca, bv_mlp_pca = tune_mlp(
    X_tr_pca, X_va_pca, X_te_pca,
    arch_list=[(16, 8), (32, 16)], dropouts=[0.25, 0.4], lrs=[1e-3, 2e-3], wds=[0.01, 0.05], patience=20
)
print(f"[PCA-{best_pca_k}] Best Compact MLP Params (Val Loss={bv_mlp_pca:.5f}): {cfg_mlp_pca}")

best_a_pca, bv_rpca, ridge_pca = None, float("inf"), None
for a in [50.0, 200.0, 500.0, 1000.0, 2000.0]:
    cand = Ridge(alpha=a, random_state=SEED).fit(X_tr_pca, y_train_s)
    vmse = mean_squared_error(y_val, cand.predict(X_va_pca) * std_y + mu_y)
    if vmse < bv_rpca:
        bv_rpca, best_a_pca, ridge_pca = vmse, a, cand
pred_ridge_pca_va = ridge_pca.predict(X_va_pca) * std_y + mu_y
pred_ridge_pca = ridge_pca.predict(X_te_pca) * std_y + mu_y
print(f"[PCA-{best_pca_k}] Best Ridge alpha={best_a_pca} (Val MSE={bv_rpca:.8f})")

best_rf_cfg, bv_rf, rf_pca = None, float("inf"), None
for ne in [100, 200]:
    for md in [3, 5]:
        for msl in [5, 10]:
            cand = RandomForestRegressor(n_estimators=ne, max_depth=md, min_samples_leaf=msl, random_state=SEED, n_jobs=-1).fit(X_tr_pca, y_train_s)
            vmse = mean_squared_error(y_val, cand.predict(X_va_pca) * std_y + mu_y)
            if vmse < bv_rf:
                bv_rf, best_rf_cfg, rf_pca = vmse, {"n_estimators": ne, "max_depth": md, "min_samples_leaf": msl}, cand
pred_rf_pca = rf_pca.predict(X_te_pca) * std_y + mu_y
print(f"[PCA-{best_pca_k}] Best RF Params (Val MSE={bv_rf:.8f}): {best_rf_cfg}")

best_gb_cfg, bv_gb, gbr_pca = None, float("inf"), None
for ne in [40, 80, 120]:
    for md in [2, 3]:
        for lr_g in [0.02, 0.05]:
            cand = GradientBoostingRegressor(n_estimators=ne, max_depth=md, learning_rate=lr_g, subsample=0.8, random_state=SEED).fit(X_tr_pca, y_train_s)
            vmse = mean_squared_error(y_val, cand.predict(X_va_pca) * std_y + mu_y)
            if vmse < bv_gb:
                bv_gb, best_gb_cfg, gbr_pca = vmse, {"n_estimators": ne, "max_depth": md, "learning_rate": lr_g}, cand
pred_gbr_pca = gbr_pca.predict(X_te_pca) * std_y + mu_y
print(f"[PCA-{best_pca_k}] Best GBDT Params (Val MSE={bv_gb:.8f}): {best_gb_cfg}")

# Tuning bobot Ensemble Blend (Compact MLP PCA + Ridge PCA)
best_wb, bv_blend = 0.5, float("inf")
for wb in [0.0, 0.25, 0.5, 0.75, 1.0]:
    vmse = mean_squared_error(y_val, wb * pred_mlp_pca_va + (1.0 - wb) * pred_ridge_pca_va)
    if vmse < bv_blend:
        bv_blend, best_wb = vmse, wb
pred_blend_pca = best_wb * pred_mlp_pca + (1.0 - best_wb) * pred_ridge_pca
print(f"[PCA-{best_pca_k}] Best Ensemble Weight: MLP={best_wb:.2f}, Ridge={1-best_wb:.2f} (Val MSE={bv_blend:.8f})")

results = [
    {"Model": "Baseline: Tebak Return 0",                        **compute_metrics(y_test, np.zeros_like(y_test), mu_y)},
    {"Model": "Baseline: Tebak Rata-rata Train",                 **compute_metrics(y_test, np.full_like(y_test, mu_y), mu_y)},
    {"Model": "[Direct 1536-d] Deep MLP (Tuned)",                **compute_metrics(y_test, pred_mlp_direct, mu_y)},
    {"Model": f"[Direct 1536-d] Ridge (alpha={best_a_1536:.0f})", **compute_metrics(y_test, pred_ridge_direct, mu_y)},
    {"Model": f"[PCA-{best_pca_k}] Compact MLP (Tuned)",         **compute_metrics(y_test, pred_mlp_pca, mu_y)},
    {"Model": f"[PCA-{best_pca_k}] Ridge (alpha={best_a_pca:.0f})", **compute_metrics(y_test, pred_ridge_pca, mu_y)},
    {"Model": f"[PCA-{best_pca_k}] Random Forest (Tuned)",       **compute_metrics(y_test, pred_rf_pca, mu_y)},
    {"Model": f"[PCA-{best_pca_k}] GBDT (Tuned)",                **compute_metrics(y_test, pred_gbr_pca, mu_y)},
    {"Model": f"[PCA-{best_pca_k}] Ensemble (w={best_wb:.2f})",  **compute_metrics(y_test, pred_blend_pca, mu_y)},
]

eval_df = pd.DataFrame(results).set_index("Model")
print("\n===================== EVALUASI DIRECT EMBEDDING CONCAT + HYPERPARAMETER TUNING =====================")
display(eval_df.round(6))""")

# ----------------- CELL 13 & 14: VISUALIZATION -----------------
add_md(r"""## 6. Visualisasi Komprehensif (4 Panel Evaluasi & Kurva Early Stopping)""")

add_code(r"""fig, axes = plt.subplots(2, 2, figsize=(18, 11), dpi=140)
eval_plot = eval_df.drop(["Baseline: Tebak Return 0", "Baseline: Tebak Rata-rata Train"])
x_idx = np.arange(len(eval_plot))
w = 0.35

# Panel 1: RMSE & MAE
ax1 = axes[0, 0]
ax1.bar(x_idx - w/2, eval_plot["RMSE"], w, label="RMSE", color="#2a9d8f")
ax1.bar(x_idx + w/2, eval_plot["MAE"], w, label="MAE", color="#e76f51")
ax1.axhline(eval_df.loc["Baseline: Tebak Rata-rata Train", "RMSE"], color="red", linestyle="--", alpha=0.75, label="Baseline Mean RMSE")
ax1.set_title("Perbandingan Galat Prediksi (RMSE & MAE) — Test Set", fontweight="bold")
ax1.set_xticks(x_idx)
ax1.set_xticklabels(eval_plot.index, rotation=25, ha="right", fontsize=8.5)
ax1.set_ylabel("Nilai Galat")
ax1.legend(fontsize=8.5)
ax1.grid(axis="y", linestyle=":", alpha=0.6)

# Panel 2: DirAcc & Balanced Accuracy
ax2 = axes[0, 1]
ax2.bar(x_idx - w/2, eval_plot["DirAcc (%)"], w, label="DirAcc (%)", color="#457b9d")
ax2.bar(x_idx + w/2, eval_plot["BalAcc (%)"], w, label="Balanced Acc (%)", color="#f4a261")
ax2.axhline(50.0, color="gray", linestyle="--", label="Tebak Koin (50%)")
ax2.set_title("Akurasi Arah: Directional vs Balanced Accuracy (%)", fontweight="bold")
ax2.set_xticks(x_idx)
ax2.set_xticklabels(eval_plot.index, rotation=25, ha="right", fontsize=8.5)
ax2.set_ylabel("Persentase (%)")
ax2.set_ylim(40, 68)
ax2.legend(fontsize=8.5)
ax2.grid(axis="y", linestyle=":", alpha=0.6)

# Panel 3: Time Series Actual vs Ensemble Blend
ax3 = axes[1, 0]
test_dates = df_valid[date_col].iloc[idx_test]
ax3.plot(test_dates, y_test, label="Log Return Aktual", color="black", alpha=0.65, linewidth=1.1)
ax3.plot(test_dates, pred_blend_pca, label="Prediksi Ensemble (Tuned)", color="#e63946", linestyle="--", linewidth=1.4)
ax3.axhline(0, color="gray", linewidth=0.8)
ax3.set_title("Deret Waktu Log Return Test Set: Aktual vs Prediksi", fontweight="bold")
ax3.set_ylabel("Log Return")
ax3.legend(fontsize=8.5)
ax3.grid(True, linestyle=":", alpha=0.6)

# Panel 4: Learning Curve (Compact MLP PCA dengan Early Stopping)
ax4 = axes[1, 1]
ax4.plot(tr_hist_pca, label="Training MSE Loss", color="#1d3557", linewidth=1.5)
ax4.plot(va_hist_pca, label="Validation MSE Loss", color="#e63946", linestyle="--", linewidth=1.5)
ax4.set_title(f"Kurva Pembelajaran Compact MLP Tuned (Berhenti di Epoch {len(va_hist_pca)})", fontweight="bold")
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Scaled MSE Loss")
ax4.legend(fontsize=9)
ax4.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()""")

# ----------------- CELL 15 & 16: SAVE REPORT -----------------
add_md(r"""## 7. Penyimpanan Hasil Evaluasi CSV""")

add_code(r"""reports_dir = Path("percobaan task 3/reports")
reports_dir.mkdir(parents=True, exist_ok=True)
out_csv = reports_dir / "evaluasi_direct_concat_embedding.csv"
eval_df.to_csv(out_csv)
print(f"Hasil evaluasi berhasil disimpan ke: {out_csv}")""")

nb_dict = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python", "version": "3.11"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = "percobaan task 3/trial4_direct_concat_embedding.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb_dict, f, indent=1, ensure_ascii=False)
print(f"Notebook Trial 4 (Freeze Weights + No Cache + Tuning) berhasil diperbarui: {out_path} ({len(cells)} sel)")
