import json
import ast

def set_src(cell, txt):
    cell["source"] = [line + "\n" for line in txt.strip().split("\n")]
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

# ===================== 1. UPDATE percobaan task 3/trial3_pipeline.ipynb =====================
p3 = "percobaan task 3/trial3_pipeline.ipynb"
with open(p3, "r", encoding="utf-8") as f:
    nb3 = json.load(f)

set_src(nb3["cells"][8], r"""# Ekstraksi fitur topik menggunakan BERTopic (Freeze Weights, Tanpa Cache Disk)
tr_filter = df_raw[date_col] <= train_end
train_docs = (df_raw.loc[tr_filter, "title_clean"] + " " + df_raw.loc[tr_filter, "fulltext_clean"]).tolist()
all_docs   = (df_raw["title_clean"] + " " + df_raw["fulltext_clean"]).tolist()

print("Memulai ekstraksi fitur topik BERTopic (in-memory, tanpa cache, freeze weights)...")
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

    # Bekukan bobot SentenceTransformer agar inferensi embedding cepat
    for param in st_model.parameters():
        param.requires_grad = False
    st_model.eval()

    topic_model = BERTopic(
        embedding_model=st_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=5,
        calculate_probabilities=True,
        verbose=False
    )
    with torch.no_grad():
        topic_model.fit(train_docs_clean)
        _, probs = topic_model.transform(all_docs_clean)

    if probs is None or len(probs.shape) == 1:
        probs = np.zeros((len(all_docs), 5))
    elif probs.shape[1] < 5:
        probs = np.pad(probs, ((0, 0), (0, 5 - probs.shape[1])))
    else:
        probs = probs[:, :5]

    print("BERTopic berhasil mengekstrak 5 probabilitas topik secara langsung!")
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

set_src(nb3["cells"][10], r"""# Inferensi FinBERT langsung di memori (Freeze Weights, Tanpa Cache Disk)
all_individual_titles = []
for raw_t in df_raw[title_col].fillna(""):
    for part in str(raw_t).split(" | "):
        t_c = clean_text_fn(part)
        if t_c and t_c.lower() != "nan":
            all_individual_titles.append(t_c)

uniq_titles = sorted(list(set(all_individual_titles)), key=len)
print(f"Total kemunculan artikel: {len(all_individual_titles)} | Judul unik: {len(uniq_titles)}")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
print(f"Menjalankan inferensi FinBERT (Freeze Weights, No Cache) untuk {len(uniq_titles)} judul unik...")
fin_tok = AutoTokenizer.from_pretrained(FINBERT_MODEL)
fin_mod = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL).to(DEVICE)

for param in fin_mod.parameters():
    param.requires_grad = False
fin_mod.eval()

fin_map = {}
bs = 64
with torch.no_grad():
    for i in range(0, len(uniq_titles), bs):
        b = uniq_titles[i:i+bs]
        inp = fin_tok(b, padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
        probs_t = torch.softmax(fin_mod(**inp).logits, dim=-1).cpu().numpy()
        for title_str, p_arr in zip(b, probs_t):
            # ProsusAI/finbert: 0->positive, 1->negative, 2->neutral
            fin_map[title_str] = {
                "finbert_pos": float(p_arr[0]),
                "finbert_neg": float(p_arr[1]),
                "finbert_neu": float(p_arr[2])
            }

del fin_mod, fin_tok
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print(f"Inferensi FinBERT selesai di memori untuk {len(fin_map)} judul unik.")""")

set_src(nb3["cells"][14], r"""# Ekstraksi Dual-Encoder Embedding langsung di memori (Freeze Weights, Tanpa Cache Disk)
print("Mengekstrak embedding teks menggunakan Dual-Encoder (Freeze Weights, No Cache)...")
from transformers import AutoTokenizer, AutoModel

def extract_cls(texts, model_name, max_len=128):
    tok = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModel.from_pretrained(model_name).to(DEVICE)
    for param in mod.parameters():
        param.requires_grad = False
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
    del mod, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.vstack(embs)

emb_title = extract_cls(df_raw["title_clean"].tolist(), TITLE_BERT_MODEL, max_len=128)
try:
    emb_body = extract_cls(df_raw["fulltext_clean"].tolist(), FULLTEXT_MODEL, max_len=1024)
except Exception as e:
    print(f"Fallback Longformer ke BERT (karena: {e})")
    emb_body = extract_cls(df_raw["fulltext_clean"].tolist(), TITLE_BERT_MODEL, max_len=512)

X_concat_all = np.hstack([emb_title, emb_body])
print(f"Dual-encoder embeddings berhasil diekstrak: shape {X_concat_all.shape}")

daily_emb = pd.DataFrame(X_concat_all).groupby(df_raw[date_col]).mean().sort_index()
train_dates_emb = daily_emb.index <= train_end
pca = PCA(n_components=6, random_state=SEED)
pca.fit(daily_emb[train_dates_emb])
pca_features = pd.DataFrame(pca.transform(daily_emb), index=daily_emb.index,
                            columns=[f"pca_{i}" for i in range(6)])
print(f"PCA Variance Explained (6 komponen utama): {pca.explained_variance_ratio_.sum():.4f}")""")

set_src(nb3["cells"][18], r"""from sklearn.model_selection import TimeSeriesSplit

tr_m = daily_valid.index <= train_end
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

# Hyperparameter Tuning Stage 1 GBDT dengan TimeSeriesSplit pada Train Set
tscv_s1 = TimeSeriesSplit(n_splits=4)
s1_grid = [
    {"n_estimators": ne, "max_depth": md, "learning_rate": lr, "subsample": 0.8}
    for ne in [25, 40, 60]
    for md in [2, 3]
    for lr in [0.03, 0.05]
]
best_s1_auc, best_s1_cfg = -1.0, s1_grid[0]
for cfg in s1_grid:
    fold_aucs = []
    for f_tr, f_va in tscv_s1.split(X_txt_tr):
        if len(np.unique(y_dir_tr[f_va])) < 2:
            continue
        m_cv = GradientBoostingClassifier(**cfg, random_state=SEED).fit(X_txt_tr[f_tr], y_dir_tr[f_tr])
        fold_aucs.append(roc_auc_score(y_dir_tr[f_va], m_cv.predict_proba(X_txt_tr[f_va])[:, 1]))
    mean_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.5
    if mean_auc > best_s1_auc:
        best_s1_auc = mean_auc
        best_s1_cfg = cfg

print(f"Stage 1 Best GBDT Params (TS-CV AUC={best_s1_auc:.4f}): {best_s1_cfg}")

# 5-Fold Out-of-Fold (OOF) tanpa shuffle menggunakan konfigurasi terbaik
kf = KFold(n_splits=5, shuffle=False)
oof_pred_prob = np.zeros(len(X_txt_tr))

for fold_tr, fold_val in kf.split(X_txt_tr):
    clf_fold = GradientBoostingClassifier(**best_s1_cfg, random_state=SEED)
    clf_fold.fit(X_txt_tr[fold_tr], y_dir_tr[fold_tr])
    oof_pred_prob[fold_val] = clf_fold.predict_proba(X_txt_tr[fold_val])[:, 1]

clf_full = GradientBoostingClassifier(**best_s1_cfg, random_state=SEED).fit(X_txt_tr, y_dir_tr)

daily_valid.loc[tr_m, "pred_text_prob_up"] = oof_pred_prob
daily_valid.loc[va_m, "pred_text_prob_up"] = clf_full.predict_proba(X_txt_va)[:, 1]
daily_valid.loc[te_m, "pred_text_prob_up"] = clf_full.predict_proba(X_txt_te)[:, 1]

p_te_s1 = daily_valid.loc[te_m, "pred_text_prob_up"].values
pred_te_s1 = (p_te_s1 >= 0.5).astype(int)
acc_s1 = accuracy_score(y_dir_te, pred_te_s1) * 100
bacc_s1 = balanced_accuracy_score(y_dir_te, pred_te_s1) * 100
cm_s1 = confusion_matrix(y_dir_te, pred_te_s1).ravel()

print(f"Stage 1 Klasifikasi Arah Teks (Test Set): Accuracy={acc_s1:.2f}%, Balanced Acc={bacc_s1:.2f}%, CM(TN,FP,FN,TP)={cm_s1}")""")

set_src(nb3["cells"][21], r"""## 10. Pemodelan Tahap 2: Hyperparameter Tuning Direction-Aware MLP & Ridge Benchmark (`log_return`)""")

set_src(nb3["cells"][22], r"""def compute_metrics(y_true, y_pred, y_train_ref):
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

# 1. Hyperparameter Tuning Master Ridge Regressor (pilih alpha optimal di Validation MSE)
best_alpha, best_ridge_val = 100.0, float("inf")
ridge_model = None
for a in [1.0, 10.0, 50.0, 100.0, 300.0, 1000.0]:
    cand_r = Ridge(alpha=a, random_state=SEED).fit(X_tr, y_tr_scaled)
    p_r_va = cand_r.predict(X_va) * std_y + mu_y
    v_mse = mean_squared_error(y_va, p_r_va)
    if v_mse < best_ridge_val:
        best_ridge_val = v_mse
        best_alpha = a
        ridge_model = cand_r

pred_ridge_va = ridge_model.predict(X_va) * std_y + mu_y
pred_ridge_te = ridge_model.predict(X_te) * std_y + mu_y
print(f"Best Ridge alpha={best_alpha} (Val MSE={best_ridge_val:.8f})")

# 2. Direction-Aware PyTorch MLP dengan Early Stopping + Hyperparameter Tuning
class DirAwareMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(8, hidden_dim // 2), 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_dir_aware_mlp(Xtr, ytr_s, ytr_raw, Xva, yva_s, yva_raw, Xte,
                        dir_weight=0.25, epochs=200, hidden_dim=32, dropout=0.3,
                        lr=3e-3, weight_decay=1e-2, patience=25, return_val=False):
    set_seed(SEED)
    model = DirAwareMLP(Xtr.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(ytr_s, dtype=torch.float32, device=DEVICE)
    st = torch.tensor(np.where(ytr_raw > 0, 1.0, -1.0), dtype=torch.float32, device=DEVICE)

    Xv = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(yva_s, dtype=torch.float32, device=DEVICE)
    sv = torch.tensor(np.where(yva_raw > 0, 1.0, -1.0), dtype=torch.float32, device=DEVICE)

    mu_ratio = float(mu_y / std_y)
    best_val, best_w, wait = float("inf"), None, 0
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
        p_va = model(Xv).cpu().numpy() * std_y + mu_y
        p_te = model(torch.tensor(Xte, dtype=torch.float32, device=DEVICE)).cpu().numpy() * std_y + mu_y
    if return_val:
        return p_te, p_va, best_val, tr_hist, va_hist
    return p_te, tr_hist, va_hist

# Tuning Basic MLP (dir_weight=0.0)
best_mse_val = float("inf")
best_mse_cfg = None
pred_mlp_mse_te = None
for h in [16, 32]:
    for d in [0.2, 0.4]:
        for lr in [1e-3, 3e-3]:
            for wd in [1e-2, 5e-2]:
                pte, pva, bv, _, _ = train_dir_aware_mlp(
                    X_tr, y_tr_scaled, y_tr, X_va, y_va_scaled, y_va, X_te,
                    dir_weight=0.0, hidden_dim=h, dropout=d, lr=lr, weight_decay=wd, return_val=True
                )
                if bv < best_mse_val:
                    best_mse_val = bv
                    best_mse_cfg = {"hidden_dim": h, "dropout": d, "lr": lr, "weight_decay": wd}
                    pred_mlp_mse_te = pte
print(f"Best Basic MLP Params (Val MSE={best_mse_val:.5f}): {best_mse_cfg}")

# Tuning Direction-Aware MLP (dir_weight, hidden_dim, dropout, lr, weight_decay)
best_dir_val = float("inf")
best_mlp_dir_cfg = None
pred_mlp_dir_te, pred_mlp_dir_va, train_loss_hist, val_loss_hist = None, None, None, None
for dw in [0.15, 0.25, 0.35]:
    for h in [16, 32]:
        for d in [0.25, 0.4]:
            for lr in [1e-3, 3e-3]:
                pte, pva, bv, th, vh = train_dir_aware_mlp(
                    X_tr, y_tr_scaled, y_tr, X_va, y_va_scaled, y_va, X_te,
                    dir_weight=dw, hidden_dim=h, dropout=d, lr=lr, weight_decay=1e-2, return_val=True
                )
                if bv < best_dir_val:
                    best_dir_val = bv
                    best_mlp_dir_cfg = {"dir_weight": dw, "hidden_dim": h, "dropout": d, "lr": lr, "weight_decay": 1e-2}
                    pred_mlp_dir_te, pred_mlp_dir_va, train_loss_hist, val_loss_hist = pte, pva, th, vh
print(f"Best Direction-Aware MLP Params (Val Loss={best_dir_val:.5f}): {best_mlp_dir_cfg}")

# 3. Tuning Bobot Ensemble Blend (Dir-Aware MLP + Ridge) di Validation Set
best_w_blend, best_blend_mse = 0.5, float("inf")
for wb in [0.0, 0.25, 0.5, 0.75, 1.0]:
    cand_blend_va = wb * pred_mlp_dir_va + (1.0 - wb) * pred_ridge_va
    bmse = mean_squared_error(y_va, cand_blend_va)
    if bmse < best_blend_mse:
        best_blend_mse = bmse
        best_w_blend = wb
pred_blend_te = best_w_blend * pred_mlp_dir_te + (1.0 - best_w_blend) * pred_ridge_te
print(f"Best Ensemble Weight (DirMLP={best_w_blend:.2f}, Ridge={1-best_w_blend:.2f}) | Val MSE={best_blend_mse:.8f}")

results_list = [
    {"Model": "Baseline: Tebak Return 0",                      **compute_metrics(y_te, np.zeros_like(y_te), mu_y)},
    {"Model": "Baseline: Tebak Rata-rata Train",               **compute_metrics(y_te, np.full_like(y_te, mu_y), mu_y)},
    {"Model": "Basic MLP (Pure MSE, Tuned)",                   **compute_metrics(y_te, pred_mlp_mse_te, mu_y)},
    {"Model": f"Master Ridge Regressor (alpha={best_alpha})",   **compute_metrics(y_te, pred_ridge_te, mu_y)},
    {"Model": "Direction-Aware MLP (Tuned)",                   **compute_metrics(y_te, pred_mlp_dir_te, mu_y)},
    {"Model": f"Ensemble Blend (w={best_w_blend:.2f})",        **compute_metrics(y_te, pred_blend_te, mu_y)}
]

eval_df = pd.DataFrame(results_list).set_index("Model")
print("\n=================================== HASIL EVALUASI PREDIKSI LOG_RETURN (TEST SET) ===================================")
display(eval_df.round(6))""")

set_src(nb3["cells"][24], r"""ablation_steps = [
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

    p_a, _, _ = train_dir_aware_mlp(
        X_tr_a, y_tr_scaled, y_tr, X_va_a, y_va_scaled, y_va, X_te_a, **best_mlp_dir_cfg
    )
    ablation_rows.append({"Tahapan Fitur": label, **compute_metrics(y_te, p_a, mu_y)})

ablation_df = pd.DataFrame(ablation_rows).set_index("Tahapan Fitur")
print("\n=================================== HASIL STUDI ABLASI (TEST SET) ===================================")
display(ablation_df.round(6))""")

for i, c in enumerate(nb3["cells"]):
    if c["cell_type"] == "code":
        clean_lines = [l for l in c["source"] if not l.lstrip().startswith(("!", "%"))]
        ast.parse("".join(clean_lines))
with open(p3, "w", encoding="utf-8") as f:
    json.dump(nb3, f, indent=1, ensure_ascii=False)
print(f"Updated {p3} cleanly ({len(nb3['cells'])} cells)")

# ===================== 2. UPDATE notebook/topic_modeling_lda_vs_bertopic_umap.ipynb =====================
pt = "notebook/topic_modeling_lda_vs_bertopic_umap.ipynb"
with open(pt, "r", encoding="utf-8") as f:
    nbt = json.load(f)

set_src(nbt["cells"][4], r"""DATA_CANDIDATES = [
    Path("data/processed/news_processed_unmerged.csv"),
    Path("../data/processed/news_processed_unmerged.csv"),
]
DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), None)
if DATA_PATH is None:
    raise FileNotFoundError(
        "Tidak menemukan data/processed/news_processed_unmerged.csv. "
        "Pastikan notebook dijalankan dari root proyek atau folder notebook/."
    )

df_raw = pd.read_csv(DATA_PATH)
print(f"Dataset dimuat dari : {DATA_PATH}")
print(f"Jumlah baris awal   : {len(df_raw):,}")
print(f"Kolom tersedia      : {list(df_raw.columns)}")

# Bersihkan dan deduplikasi artikel
df = (
    df_raw.dropna(subset=["title", "text_lemmatized", "normalized_text"])
    .drop_duplicates(subset=["text_lemmatized"])
    .reset_index(drop=True)
)

df["date"] = pd.to_datetime(df["tanggal"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
df["year_quarter"] = df["date"].dt.to_period("Q").astype(str)
df["word_count"] = df["text_lemmatized"].astype(str).str.split().str.len()

print(f"Jumlah artikel unik : {len(df):,}")
print(f"Rentang tanggal     : {df['date'].min().date()} s/d {df['date'].max().date()}")
print(f"Rata-rata kata/art. : {df['word_count'].mean():.1f} (median: {df['word_count'].median():.0f})")

df[["date", "title", "word_count"]].head(5)""")

set_src(nbt["cells"][13], r"""import torch
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# Bekukan seluruh bobot encoder (Freeze Weights) agar ekstraksi cepat dan bebas fine-tuning
for param in embedding_model.parameters():
    param.requires_grad = False
embedding_model.eval()

print(f"Menghitung embedding {len(docs_semantic):,} artikel dengan all-MiniLM-L6-v2 (Freeze Weights, No Cache)...")
with torch.no_grad():
    embeddings = embedding_model.encode(docs_semantic, show_progress_bar=True, batch_size=64, normalize_embeddings=True)
print(f"Embedding selesai dihitung di memori: {embeddings.shape}")""")

set_src(nbt["cells"][14], r"""# Hyperparameter Tuning BERTopic (min_cluster_size & n_neighbors) berdasarkan Coherence (c_v)
vectorizer_model = CountVectorizer(
    stop_words=sorted(COMBINED_STOPWORDS),
    min_df=5,
    ngram_range=(1, 2),
)
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
representation_model = MaximalMarginalRelevance(diversity=0.35)

analyzer_bt = vectorizer_model.build_analyzer()
tokens_bt = [analyzer_bt(d) for d in docs_bow]
dict_bt = Dictionary(tokens_bt)

bt_tune_rows = []
best_bt_score = -1e9
best_bt_cfg = {"n_neighbors": 15, "min_cluster_size": 35}

for nn_val, mcs_val in [(10, 25), (15, 35), (15, 50), (25, 35)]:
    u_cand = UMAP(n_neighbors=nn_val, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED)
    h_cand = HDBSCAN(min_cluster_size=mcs_val, min_samples=10, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
    tm_cand = BERTopic(
        embedding_model=embedding_model,
        umap_model=u_cand,
        hdbscan_model=h_cand,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        top_n_words=10,
        verbose=False,
    )
    t_cand, _ = tm_cand.fit_transform(docs_bow, embeddings)
    t_ids = [t for t in tm_cand.get_topics() if t != -1]
    out_r = float((np.array(t_cand) == -1).mean())
    if len(t_ids) >= 2:
        t_w = [[w for w, _ in tm_cand.get_topic(t)] for t in t_ids]
        coh = float(CoherenceModel(topics=t_w, texts=tokens_bt, dictionary=dict_bt, coherence="c_v", processes=1).get_coherence())
    else:
        coh = 0.0
    sc = coh - 0.15 * out_r
    bt_tune_rows.append({"n_neighbors": nn_val, "min_cluster_size": mcs_val, "n_topics": len(t_ids), "outlier_pct": round(out_r * 100, 2), "coherence_cv": round(coh, 4), "score": round(sc, 4)})
    if sc > best_bt_score:
        best_bt_score = sc
        best_bt_cfg = {"n_neighbors": nn_val, "min_cluster_size": mcs_val}

bt_tune_df = pd.DataFrame(bt_tune_rows).sort_values("score", ascending=False).reset_index(drop=True)
print("Hasil Tuning BERTopic:")
display(bt_tune_df)
print("Konfigurasi BERTopic terbaik:", best_bt_cfg)

umap_5d = UMAP(
    n_neighbors=best_bt_cfg["n_neighbors"],
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=SEED,
)
hdbscan_model = HDBSCAN(
    min_cluster_size=best_bt_cfg["min_cluster_size"],
    min_samples=10,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_5d,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    representation_model=representation_model,
    top_n_words=10,
    verbose=False,
)

topics_raw, probs_raw = topic_model.fit_transform(docs_bow, embeddings)
topics_raw = np.array(topics_raw)

n_topics_raw = len([t for t in set(topics_raw) if t != -1])
n_out_raw = int((topics_raw == -1).sum())
print(f"[Sebelum Reduksi Outlier] Topik: {n_topics_raw} | Outlier (-1): {n_out_raw:,} ({n_out_raw/len(df)*100:.1f}%)")

if n_out_raw > 0:
    topics_reduced = np.array(
        topic_model.reduce_outliers(
            docs_bow,
            topics_raw.tolist(),
            strategy="embeddings",
            embeddings=embeddings,
            threshold=0.35,
        )
    )
    topic_model.update_topics(
        docs_bow,
        topics=topics_reduced.tolist(),
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
    )
else:
    topics_reduced = topics_raw.copy()

n_out_after = int((topics_reduced == -1).sum())
print(f"[Setelah Reduksi Outlier]  Topik: {n_topics_raw} | Outlier (-1): {n_out_after:,} ({n_out_after/len(df)*100:.1f}%)")

df["bt_topic_raw"] = topics_raw
df["bt_topic"] = topics_reduced

bt_info = topic_model.get_topic_info()
bt_label_map = {}
bt_summary_rows = []
for _, row in bt_info.iterrows():
    tid = int(row["Topic"])
    words = [w for w, _ in topic_model.get_topic(tid)]
    lbl = "BT-Outlier (-1)" if tid == -1 else f"BT-{tid}: {', '.join(words[:3])}"
    bt_label_map[tid] = lbl
    cnt = int((df["bt_topic"] == tid).sum())
    bt_summary_rows.append({
        "Topic_ID": tid,
        "Label": lbl,
        "Count": cnt,
        "Share (%)": round(cnt / len(df) * 100, 2),
        "Top_10_Words": ", ".join(words),
    })

df["bt_label"] = df["bt_topic"].map(bt_label_map)
df["bt_label_raw"] = df["bt_topic_raw"].map(lambda t: "Outlier (-1)" if t == -1 else bt_label_map.get(t, f"BT-{t}"))
bt_summary_df = pd.DataFrame(bt_summary_rows).sort_values("Count", ascending=False).reset_index(drop=True)
bt_summary_df.head(20)""")

for i, c in enumerate(nbt["cells"]):
    if c["cell_type"] == "code":
        clean_lines = [l for l in c["source"] if not l.lstrip().startswith(("!", "%"))]
        ast.parse("".join(clean_lines))
with open(pt, "w", encoding="utf-8") as f:
    json.dump(nbt, f, indent=1, ensure_ascii=False)
print(f"Updated {pt} cleanly ({len(nbt['cells'])} cells)")
