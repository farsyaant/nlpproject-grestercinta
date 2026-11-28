import json
import ast

# ===================== 1. UPDATE trial1.ipynb =====================
p1 = "percobaan task 3/trial1.ipynb"
with open(p1, "r", encoding="utf-8") as f:
    nb1 = json.load(f)

def set_src(cell, txt):
    cell["source"] = [line + "\n" for line in txt.strip().split("\n")]
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

set_src(nb1["cells"][4], r"""## 3. Ekstraksi Embeddings Transformer (FinBERT, BERT, RoBERTa) — Freeze Weights & Tanpa Cache
Seluruh bobot model pretrained **dibekukan (`requires_grad = False`)** agar ekstraksi berjalan cepat di mode inferensi (`eval()` + `torch.no_grad()`) dan **dihitung langsung di memori tanpa menyimpan/memuat cache disk**.""")

set_src(nb1["cells"][5], r"""MODEL_CONFIGS = {
    "finbert": {"name": "ProsusAI/finbert", "dim": 768},
    "bert": {"name": "bert-base-uncased", "dim": 768},
    "roberta": {"name": "roberta-base", "dim": 768}
}

def extract_embeddings(texts, model_name, desc_tag=None, batch_size=32, max_length=128, device=DEVICE):
    print(f"Mengekstrak embedding (Freeze Weights, No Cache) dengan '{model_name}' (max_length={max_length})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    # Bekukan seluruh bobot encoder pretrained agar inferensi cepat
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    all_embeddings = []
    text_list = [str(t) if pd.notna(t) and str(t).strip() else "empty news" for t in texts]

    with torch.no_grad():
        for i in tqdm(range(0, len(text_list), batch_size), desc=f"Embedding ({desc_tag or model_name})"):
            batch_texts = text_list[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            outputs = model(**encoded)
            last_hidden = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            sum_embeddings = torch.sum(last_hidden * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_pooled = (sum_embeddings / sum_mask).cpu().numpy()
            all_embeddings.append(mean_pooled)

    X_emb = np.vstack(all_embeddings).astype(np.float32)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return X_emb""")

set_src(nb1["cells"][6], r"""# Ekstraksi Embeddings untuk Title (max_length=64) dan Full-Text (max_length=128) tanpa cache
embeddings_dict = {}

for model_tag, cfg in MODEL_CONFIGS.items():
    print("=" * 70)
    print(f"MODEL: {model_tag.upper()} ({cfg['name']})")
    print("=" * 70)

    emb_title = extract_embeddings(
        texts=news_df["title"],
        model_name=cfg["name"],
        desc_tag=f"{model_tag}_title",
        batch_size=64,
        max_length=64
    )
    embeddings_dict[f"{model_tag}_title"] = emb_title

    emb_fulltext = extract_embeddings(
        texts=news_df["full_text"],
        model_name=cfg["name"],
        desc_tag=f"{model_tag}_fulltext",
        batch_size=32,
        max_length=128
    )
    embeddings_dict[f"{model_tag}_fulltext"] = emb_fulltext""")

set_src(nb1["cells"][10], r"""def evaluate_predictions(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    valid_dir = y_true != 0
    if np.sum(valid_dir) > 0:
        dir_acc = float(np.mean(np.sign(y_pred[valid_dir]) == np.sign(y_true[valid_dir])) * 100)
    else:
        dir_acc = 50.0

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "DirAcc (%)": dir_acc}

# Naive Baseline: Menebak selalu return 0
naive_metrics = evaluate_predictions(y_test, np.zeros_like(y_test))
print("Naive Baseline (Tebak Return 0):", naive_metrics)""")

set_src(nb1["cells"][13], r"""def train_mlp_model(in_dim, X_train, y_train, X_val, y_val,
                    hidden_dim=256, dropout=0.3, epochs=70, lr=1e-3,
                    weight_decay=1e-4, batch_size=256, patience=12):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = DeepMLPRegressor(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_loss = float("inf")
    best_weights = None
    patience_cnt = 0

    X_tr_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    n_tr = X_tr_t.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        for start in range(0, n_tr, batch_size):
            idx = perm[start:start + batch_size]
            xb = X_tr_t[idx]
            if xb.shape[0] < 2:
                continue
            optimizer.zero_grad()
            loss = criterion(model(xb), y_tr_t[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_weights:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_weights.items()})

    return model, best_loss

def tune_mlp_hyperparams(in_dim, X_train, y_train, X_val, y_val):
    param_grid = [
        {"hidden_dim": h, "dropout": d, "lr": lr, "weight_decay": wd}
        for h in [64, 128, 256]
        for d in [0.3, 0.5]
        for lr in [5e-4, 1e-3]
        for wd in [1e-3]
    ]
    best_score = float("inf")
    best_cfg = None
    best_model = None
    for cfg in param_grid:
        cand_model, val_loss = train_mlp_model(
            in_dim=in_dim, X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val, epochs=50, patience=10, **cfg
        )
        if val_loss < best_score:
            best_score = val_loss
            best_cfg = cfg
            best_model = cand_model
    return best_model, best_cfg, best_score

def predict_mlp_model(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        preds = model(X_t).cpu().numpy().ravel()
    return preds""")

set_src(nb1["cells"][14], r"""## 7. Eksekusi Benchmark & Hyperparameter Tuning MLP per Embedding""")

set_src(nb1["cells"][15], r"""def model_tag_format(name):
    return {"finbert": "FinBERT", "bert": "BERT", "roberta": "RoBERTa"}.get(name, name)

print("Menjalankan Hyperparameter Tuning & Benchmark Embedding menggunakan Model MLP...")
benchmark_results = []

for key, X_all in daily_datasets.items():
    model_name, text_type = key.split("_")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_all[idx_train])
    X_val = scaler.transform(X_all[idx_val])
    X_test = scaler.transform(X_all[idx_test])

    trained_mlp, best_cfg, best_val_mse = tune_mlp_hyperparams(
        in_dim=768, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val
    )
    print(f"  [{key}] Best Params: {best_cfg} | Best Val MSE: {best_val_mse:.8f}")

    preds_val = predict_mlp_model(trained_mlp, X_val)
    preds_test = predict_mlp_model(trained_mlp, X_test)

    val_metrics = evaluate_predictions(y_val, preds_val)
    test_metrics = evaluate_predictions(y_test, preds_test)

    benchmark_results.append({
        "Model": "Deep MLP (Tuned)",
        "Embedding": model_tag_format(model_name),
        "Sumber Teks": text_type.capitalize(),
        "Best Params": str(best_cfg),
        "Val MAE": val_metrics["MAE"],
        "Val MSE": val_metrics["MSE"],
        "Test MAE": test_metrics["MAE"],
        "Test MSE": test_metrics["MSE"],
        "Test RMSE": test_metrics["RMSE"],
        "Test DirAcc (%)": test_metrics["DirAcc (%)"]
    })

benchmark_df = pd.DataFrame(benchmark_results)
benchmark_df = benchmark_df.sort_values("Test DirAcc (%)", ascending=False).reset_index(drop=True)

print("=" * 75)
print("HASIL BENCHMARK EMBEDDING DENGAN HYPERPARAMETER TUNING MLP")
print("=" * 75)
display(benchmark_df)""")

for i, c in enumerate(nb1["cells"]):
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
with open(p1, "w", encoding="utf-8") as f:
    json.dump(nb1, f, indent=1, ensure_ascii=False)
print(f"Updated {p1} cleanly ({len(nb1['cells'])} cells)")

# ===================== 2. UPDATE trial2_bertopic.ipynb =====================
p2 = "percobaan task 3/trial2_bertopic.ipynb"
with open(p2, "r", encoding="utf-8") as f:
    nb2 = json.load(f)

set_src(nb2["cells"][7], r"""## 3. Embedding (Freeze Weights, Tanpa Cache Disk)""")
set_src(nb2["cells"][8], r"""import torch
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
for param in embedding_model.parameters():
    param.requires_grad = False
embedding_model.eval()

with torch.no_grad():
    embeddings = embedding_model.encode(docs, show_progress_bar=True, batch_size=64)
print("Embeddings shape (in-memory, no cache):", embeddings.shape)""")

set_src(nb2["cells"][9], r"""## 4. Konfigurasi & Hyperparameter Tuning BERTopic (`min_cluster_size`, `n_neighbors`)""")
set_src(nb2["cells"][10], r"""try:
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False

vectorizer_model = CountVectorizer(stop_words="english", min_df=3, ngram_range=(1, 2))
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
representation_model = MaximalMarginalRelevance(diversity=0.3)

# Gunakan matriks biner sparse CountVectorizer (hemat RAM, bebas MemoryError)
X_coh_bin = (vectorizer_model.fit_transform(docs) > 0).tocsc()
voc_coh = vectorizer_model.vocabulary_
n_docs_coh = max(X_coh_bin.shape[0], 1)

def compute_coherence_score(topic_words_list):
    if not topic_words_list:
        return 0.0
    if HAS_GENSIM:
        analyzer = vectorizer_model.build_analyzer()
        tokens_tune = [analyzer(d) for d in docs]
        dictionary_tune = Dictionary(tokens_tune)
        return float(CoherenceModel(topics=topic_words_list, texts=tokens_tune, dictionary=dictionary_tune, coherence="c_v", processes=1).get_coherence())
    scores = []
    for ws in topic_words_list:
        cols = [voc_coh[w] for w in ws[:10] if w in voc_coh]
        for i in range(len(cols)):
            ci = X_coh_bin[:, cols[i]]
            d_i = int(ci.nnz)
            if d_i == 0:
                continue
            for j in range(i + 1, len(cols)):
                cj = X_coh_bin[:, cols[j]]
                d_j = int(cj.nnz)
                if d_j == 0:
                    continue
                d_ij = int(ci.multiply(cj).nnz)
                if d_ij > 0:
                    pmi = np.log((d_ij * n_docs_coh) / (d_i * d_j))
                    npmi = pmi / (-np.log(d_ij / n_docs_coh) + 1e-9)
                    scores.append(npmi)
    return float(np.mean(scores)) if scores else 0.0

mcs_candidates = sorted(list({max(5, len(docs) // 300), max(8, len(docs) // 200), max(12, len(docs) // 120)}))
cand_pairs = [(10, mcs_candidates[0]), (15, mcs_candidates[len(mcs_candidates)//2]), (15, mcs_candidates[-1]), (20, mcs_candidates[len(mcs_candidates)//2])]
cand_pairs = list(dict.fromkeys(cand_pairs))

tuning_logs = []
best_score = -1e9
best_cfg = {"n_neighbors": 15, "min_cluster_size": mcs_candidates[0]}

for nn_val, mcs in cand_pairs:
    umap_cand = UMAP(n_neighbors=nn_val, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED)
    hdb_cand = HDBSCAN(min_cluster_size=mcs, min_samples=max(3, mcs // 2), metric="euclidean", cluster_selection_method="eom", prediction_data=True)
    tm_cand = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_cand,
        hdbscan_model=hdb_cand,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        top_n_words=10,
        verbose=False,
    )
    t_cand, _ = tm_cand.fit_transform(docs, embeddings)
    t_ids = [t for t in tm_cand.get_topics() if t != -1]
    out_ratio = float((np.array(t_cand) == -1).mean())
    if len(t_ids) >= 2:
        t_words = [[w for w, _ in tm_cand.get_topic(t)] for t in t_ids]
        coh = compute_coherence_score(t_words)
    else:
        coh = 0.0
    score = coh - 0.15 * out_ratio
    tuning_logs.append({"n_neighbors": nn_val, "min_cluster_size": mcs, "n_topics": len(t_ids), "outlier_pct": round(out_ratio * 100, 2), "coherence": round(coh, 4), "tune_score": round(score, 4)})
    if score > best_score:
        best_score = score
        best_cfg = {"n_neighbors": nn_val, "min_cluster_size": mcs}

tuning_df = pd.DataFrame(tuning_logs).sort_values("tune_score", ascending=False).reset_index(drop=True)
print("Hasil Hyperparameter Tuning BERTopic:")
display(tuning_df)
print("Best Hyperparameters:", best_cfg)

min_cluster_size = best_cfg["min_cluster_size"]
umap_model = UMAP(n_neighbors=best_cfg["n_neighbors"], n_components=5, min_dist=0.0, metric="cosine", random_state=SEED)
hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=max(3, min_cluster_size // 2), metric="euclidean", cluster_selection_method="eom", prediction_data=True)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    representation_model=representation_model,
    top_n_words=10,
    verbose=True,
)""")

set_src(nb2["cells"][16], r"""topic_ids = [t for t in topic_model.get_topics() if t != -1]
topic_words = [[w for w, _ in topic_model.get_topic(t)] for t in topic_ids]

coherence = compute_coherence_score(topic_words)
diversity = len({w for ws in topic_words for w in ws}) / max(sum(len(ws) for ws in topic_words), 1)
lbl_coh = "Coherence (c_v)" if HAS_GENSIM else "Coherence (NPMI)"

print(f"{lbl_coh}: {coherence:.3f}")
print(f"Topic diversity: {diversity:.3f}   (mendekati 1 = topik tidak tumpang tindih)")

# Inspeksi manual: 3 dokumen representatif per topik
for t in topic_ids[:10]:
    words = ", ".join(w for w, _ in topic_model.get_topic(t)[:6])
    print(f"\n=== Topik {t} | {words}")
    for d in topic_model.get_representative_docs(t)[:3]:
        print("  -", d[:160].replace("\n", " "))""")

for i, c in enumerate(nb2["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        if src.lstrip().startswith("%pip"):
            continue
        ast.parse(src)
with open(p2, "w", encoding="utf-8") as f:
    json.dump(nb2, f, indent=1, ensure_ascii=False)
print(f"Updated {p2} cleanly ({len(nb2['cells'])} cells)")
