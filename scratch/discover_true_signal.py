import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix

# 1. Load data & check timezone/timestamps
unm = pd.read_csv('data/processed/news_processed_unmerged.csv', usecols=['title', 'date_raw_dt', 'jam', 'aligned_trading_date'])
print("Sample timestamps in unmerged:", unm[['date_raw_dt', 'jam', 'aligned_trading_date']].head(5).to_dict('records'))

df_raw = pd.read_csv('data/processed/merged_data_clean.csv')
fin_df = pd.read_csv('data/processed/finbert_unique_titles.csv')
fin_map = fin_df.set_index('title')[['finbert_pos', 'finbert_neg', 'finbert_neu']].to_dict('index')

df_raw['Tanggal_dt'] = pd.to_datetime(df_raw['Tanggal_dt'])
df_raw = df_raw.sort_values('Tanggal_dt').reset_index(drop=True)
df_raw['rate_clean'] = (df_raw['Kurs Jual'] + df_raw['Kurs Beli']) / 2.0

# Aggregate FinBERT + Entity + GPR per day cleanly (pre-compiled regex)
US_CASE = re.compile(r"\b(U\.S\.?|USA|US|USD|Fed|FOMC|Powell|Yellen|Bessent|Treasury|Wall Street|White House|Trump|Biden)\b")
US_LOWER = re.compile(r"\b(united states|america|american|federal reserve|the fed|dollar|greenback)\b")
IDR_CASE = re.compile(r"\b(IDR|BI|IHSG|Rupiah|Indonesia|Indonesian|Jakarta|Jokowi|Prabowo|Sri Mulyani|Perry Warjiyo|Bank Indonesia)\b")
IDR_LOWER = re.compile(r"\b(indonesia|indonesian|rupiah|jakarta|bank indonesia|bi rate|bi-rate|ihsg)\b")
GPR_SET = {
    "war", "military", "conflict", "geopolitic", "geopolitical", "threat", "sanction", "sanctions",
    "tension", "tensions", "crisis", "attack", "invasion", "nuclear", "terrorism", "terrorist",
    "missile", "hostility", "clash", "rebel", "insurgency", "troops", "sovereignty", "escalation",
    "retaliation", "dispute", "standoff", "blockade", "weapon", "drills", "border", "combat", "tariff", "tariffs"
}

daily_rows = []
for idx, row in df_raw.iterrows():
    titles = str(row['all_title']).split(' | ') if pd.notna(row['all_title']) else []
    bodies = str(row['all_full_text']).split(' | ') if pd.notna(row['all_full_text']) else []
    art_feats = []
    for i, t in enumerate(titles):
        t_c = t.strip()
        if not t_c or t_c.lower() == 'nan': continue
        b_c = bodies[i].strip() if i < len(bodies) else ""
        scores = fin_map.get(t_c, {'finbert_pos': 0.5, 'finbert_neg': 0.5})
        pos, neg = scores['finbert_pos'], scores['finbert_neg']
        pol_bin = 1.0 if pos >= neg else -1.0
        pol_margin = (pos - neg) / (pos + neg + 1e-8)
        
        full_t = f"{t_c} {b_c}"
        us_title = int(bool(US_CASE.search(t_c) or US_LOWER.search(t_c.lower())))
        idr_title = int(bool(IDR_CASE.search(t_c) or IDR_LOWER.search(t_c.lower())))
        
        toks = full_t.lower().split()
        gpr_d = sum(1 for w in toks if w in GPR_SET) / max(len(toks), 1)
        
        art_feats.append({
            'pol_bin': pol_bin, 'pol_margin': pol_margin,
            'us_title': float(us_title), 'idr_title': float(idr_title),
            'ent_title_dir': us_title * pol_bin - idr_title * 2.0 * pol_bin,
            'gpr_d': gpr_d,
            'gpr_weak': gpr_d * (-pol_bin),
        })
    if art_feats:
        agg = pd.DataFrame(art_feats).mean().to_dict()
        agg['has_news'] = 1.0
    else:
        agg = {'pol_bin': 0.0, 'pol_margin': 0.0, 'us_title': 0.0, 'idr_title': 0.0,
               'ent_title_dir': 0.0, 'gpr_d': 0.0, 'gpr_weak': 0.0, 'has_news': 0.0}
    agg['date'] = row['Tanggal_dt']
    agg['rate_clean'] = row['rate_clean']
    daily_rows.append(agg)

daily = pd.DataFrame(daily_rows).set_index('date').sort_index()
daily['log_ret'] = np.log(daily['rate_clean'] / daily['rate_clean'].shift(1))
daily['log_ret_next'] = daily['log_ret'].shift(-1)

# Check correlation of text features with SAME-DAY log_ret vs NEXT-DAY log_ret (t+1)!
n = len(daily)
n_tr = int(n * 0.70)
tr_d = daily.iloc[1:n_tr]
print("\n=== KORELASI TEKS DENGAN RETURN HARI INI (t) vs BESOK (t+1) PADA TRAIN ===")
for c in ['pol_bin', 'pol_margin', 'us_title', 'idr_title', 'ent_title_dir', 'gpr_d', 'gpr_weak']:
    r0, p0 = pearsonr(tr_d[c], tr_d['log_ret'])
    d_n = tr_d.dropna(subset=['log_ret_next'])
    r1, p1 = pearsonr(d_n[c], d_n['log_ret_next'])
    print(f"  {c:15s}: same-day r={r0:+.4f} (p={p0:.3f}) | next-day(t+1) r={r1:+.4f} (p={p1:.3f})")

# Check cached BERT/FinBERT/RoBERTa embeddings on daily level!
emb_bert_t = np.load('data/cache_embeddings/bert_title.npy')
emb_fin_t  = np.load('data/cache_embeddings/finbert_title.npy')
unm_dates = pd.to_datetime(unm['aligned_trading_date']).dt.normalize()
date_to_idx = {d: i for i, d in enumerate(daily.index)}

for emb_name, emb_mat in [('BERT_Title', emb_bert_t), ('FinBERT_Title', emb_fin_t)]:
    X_day = np.zeros((len(daily), emb_mat.shape[1]), dtype=np.float32)
    for dt_val, grp in unm.groupby(unm_dates):
        if dt_val in date_to_idx:
            X_day[date_to_idx[dt_val]] = emb_mat[grp.index].mean(axis=0)
    
    # Evaluate PCA + Ridge classification on t vs t+1!
    for target_name, y_col in [('same_day(t)', 'log_ret'), ('next_day(t+1)', 'log_ret_next')]:
        valid_idx = daily[y_col].notna().values
        X_v = X_day[valid_idx]
        y_v = daily.loc[valid_idx, y_col].values
        y_dir = (y_v > 0).astype(int)
        
        nt = len(y_v)
        i_tr = int(nt * 0.70)
        i_va = int(nt * 0.85)
        
        sc = StandardScaler().fit(X_v[:i_tr])
        pca = PCA(n_components=16, random_state=42).fit(sc.transform(X_v[:i_tr]))
        Z_tr = pca.transform(sc.transform(X_v[:i_tr]))
        Z_te = pca.transform(sc.transform(X_v[i_va:]))
        
        clf = LogisticRegression(C=0.1, max_iter=500, random_state=42).fit(Z_tr, y_dir[:i_tr])
        p_te = clf.predict_proba(Z_te)[:, 1]
        acc = accuracy_score(y_dir[i_va:], (p_te >= 0.5).astype(int)) * 100
        bacc = balanced_accuracy_score(y_dir[i_va:], (p_te >= 0.5).astype(int)) * 100
        auc = roc_auc_score(y_dir[i_va:], p_te)
        
        # Also test TF-IDF Ridge / LogReg on embeddings directly (high regularization)
        clf_full = LogisticRegression(C=0.005, max_iter=500, random_state=42).fit(sc.transform(X_v[:i_tr]), y_dir[:i_tr])
        p_te_f = clf_full.predict_proba(sc.transform(X_v[i_va:]))[:, 1]
        acc_f = accuracy_score(y_dir[i_va:], (p_te_f >= 0.5).astype(int)) * 100
        bacc_f = balanced_accuracy_score(y_dir[i_va:], (p_te_f >= 0.5).astype(int)) * 100
        auc_f = roc_auc_score(y_dir[i_va:], p_te_f)
        print(f"{emb_name:14s} -> {target_name:14s} | PCA16: Acc={acc:.2f}% Bal={bacc:.2f}% AUC={auc:.4f} | Full768(C=0.005): Acc={acc_f:.2f}% Bal={bacc_f:.2f}% AUC={auc_f:.4f}")
