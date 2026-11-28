import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

# 1. Load daily dataset & finbert article cache
df_raw = pd.read_csv("data/processed/merged_data_clean.csv",
                     usecols=["Tanggal_dt", "Kurs Jual", "Kurs Beli", "all_title", "all_full_text"])
df_raw["Tanggal_dt"] = pd.to_datetime(df_raw["Tanggal_dt"])
df_raw = df_raw.sort_values("Tanggal_dt").reset_index(drop=True)
df_raw["rate_close"] = (df_raw["Kurs Jual"] + df_raw["Kurs Beli"]) / 2.0

fin_df = pd.read_csv("data/processed/finbert_unique_titles.csv")
fin_map = fin_df.set_index("title")[["finbert_pos", "finbert_neg", "finbert_neu"]].to_dict("index")

# 2. Label Arah Besok (d -> d+1):
# Label 0 jika harga tutup hari d+1 lebih rendah dari hari d, dan 1 jika sama atau lebih tinggi.
df_raw["rate_next"] = df_raw["rate_close"].shift(-1)
df_raw["label_dir_next"] = (df_raw["rate_next"] >= df_raw["rate_close"]).astype(int)
df_valid = df_raw.dropna(subset=["rate_next"]).reset_index(drop=True)

# 3. Pass 1 (Relevance Filtering) & Pass 2 (FX Paragraph / Sentence Extraction)
FX_MACRO_PATTERN = re.compile(
    r"\b(dollar|usd|rupiah|idr|fed|federal reserve|powell|yellen|bessent|interest rate|rate hike|rate cut|"
    r"inflation|cpi|treasury|yield|bond|tariff|tariffs|sanction|sanctions|oil|crude|opec|currency|fx|"
    r"forex|exchange rate|central bank|bank indonesia|export|import|trade balance|recession|gdp|economy)\b",
    re.IGNORECASE
)

def two_pass_filter_and_extract(titles_str, bodies_str):
    titles = str(titles_str).split(" | ") if pd.notna(titles_str) else []
    bodies = str(bodies_str).split(" | ") if pd.notna(bodies_str) else []
    
    kept_texts = []
    sent_scores = []
    for i, t in enumerate(titles):
        tc = t.strip()
        if not tc or tc.lower() == "nan":
            continue
        bc = bodies[i].strip() if i < len(bodies) else ""
        if bc.lower() == "nan":
            bc = ""
        
        full_art = f"{tc}. {bc}"
        # Pass 1: Relevance Filtering (apakah artikel membahas topik makro/FX/USD/geopolitik ekonomi?)
        if not FX_MACRO_PATTERN.search(full_art):
            continue
            
        # Pass 2: Paragraph / Sentence Extraction (ambil hanya kalimat/paragraf yang relevan FX/makro)
        sentences = re.split(r"(?<=[.!?])\s+|\n+", bc)
        rel_paras = [s.strip() for s in sentences if len(s.strip()) > 25 and FX_MACRO_PATTERN.search(s)]
        extracted = f"{tc}. " + " ".join(rel_paras[:4]) if rel_paras else tc
        kept_texts.append(extracted)
        
        sc = fin_map.get(tc, {"finbert_pos": 0.33, "finbert_neg": 0.33})
        sent_scores.append(sc["finbert_pos"] - sc["finbert_neg"])
        
    if not kept_texts:
        return "", 0.0, 0
    return " ".join(kept_texts), float(np.mean(sent_scores)), len(kept_texts)

res = [two_pass_filter_and_extract(t, b) for t, b in zip(df_valid["all_title"], df_valid["all_full_text"])]
df_valid["filtered_fx_text"] = [r[0] for r in res]
df_valid["llm_proxy_sentiment"] = [r[1] for r in res]
df_valid["kept_articles"] = [r[2] for r in res]

# Filter only trading days that have relevant filtered FX texts (like the paper removing irrelevant days)
df_fx = df_valid[df_valid["kept_articles"] > 0].reset_index(drop=True)
print(f"Total hari valid: {len(df_valid)} -> Setelah 2-Pass Filtering relevan FX: {len(df_fx)} hari ({df_fx['kept_articles'].sum()} artikel relevan)")
print(f"Distribusi Label Arah (0=Turun, 1=Sama/Naik): {(df_fx['label_dir_next']==0).sum()} vs {(df_fx['label_dir_next']==1).sum()}")

# 4. Chronological Split (70 / 15 / 15)
n = len(df_fx)
n_tr, n_va = int(n * 0.70), int(n * 0.15)
tr_m = np.arange(n) < n_tr
va_m = (np.arange(n) >= n_tr) & (np.arange(n) < n_tr + n_va)
te_m = np.arange(n) >= n_tr + n_va

y_dir = df_fx["label_dir_next"].values

# 5. LDA Topic Modeling (fit di Train set saja) sebagai Pembanding
cv = CountVectorizer(max_features=1500, stop_words="english", min_df=3, max_df=0.85)
X_bow_tr = cv.fit_transform(df_fx.loc[tr_m, "filtered_fx_text"])
X_bow_te = cv.transform(df_fx.loc[te_m, "filtered_fx_text"])

lda = LatentDirichletAllocation(n_components=8, random_state=42, max_iter=20)
X_lda_tr = lda.fit_transform(X_bow_tr)
X_lda_te = lda.transform(X_bow_te)

def eval_cls(name, y_true, prob):
    pred = (prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, pred) * 100
    bal = balanced_accuracy_score(y_true, pred) * 100
    f1  = f1_score(y_true, pred, average="macro") * 100
    auc = roc_auc_score(y_true, prob)
    print(f"  {name:35s}: Acc={acc:.2f}% | BalAcc={bal:.2f}% | MacroF1={f1:.2f}% | AUC={auc:.4f}")

print("\n=== PEMBANDING AWAL (BASELINE, LDA, SENTIMEN) PADA TEST SET ===")
# Baseline Mayoritas
maj_prob = np.full(te_m.sum(), y_dir[tr_m].mean())
eval_cls("Baseline Mayoritas (Selalu 1)", y_dir[te_m], maj_prob)

# LDA + LogReg
clf_lda = LogisticRegression(C=0.5, random_state=42).fit(X_lda_tr, y_dir[tr_m])
eval_cls("LDA (8 Topik) + LogReg", y_dir[te_m], clf_lda.predict_proba(X_lda_te)[:, 1])

# LLM Proxy Sentiment + LogReg
clf_sent = LogisticRegression(C=0.5, random_state=42).fit(df_fx.loc[tr_m, ["llm_proxy_sentiment"]], y_dir[tr_m])
eval_cls("Sentiment Score + LogReg", y_dir[te_m], clf_sent.predict_proba(df_fx.loc[te_m, ["llm_proxy_sentiment"]])[:, 1])

# LDA + Sentiment + LogReg
X_comb_tr = np.hstack([X_lda_tr, df_fx.loc[tr_m, ["llm_proxy_sentiment"]].values])
X_comb_te = np.hstack([X_lda_te, df_fx.loc[te_m, ["llm_proxy_sentiment"]].values])
clf_comb = LogisticRegression(C=0.5, random_state=42).fit(X_comb_tr, y_dir[tr_m])
eval_cls("LDA + Sentiment + LogReg", y_dir[te_m], clf_comb.predict_proba(X_comb_te)[:, 1])
