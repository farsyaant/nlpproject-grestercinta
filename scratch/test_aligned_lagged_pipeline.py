import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix

SEED = 42
np.random.seed(SEED)

# 1. Load datasets
df_raw = pd.read_csv('data/processed/merged_data_clean.csv')
fin_df = pd.read_csv('data/processed/finbert_unique_titles.csv')
fin_map = fin_df.set_index('title')[['finbert_pos', 'finbert_neg', 'finbert_neu']].to_dict('index')
unm = pd.read_csv('data/processed/news_processed_unmerged.csv', usecols=['aligned_trading_date'])
emb_fin_t = np.load('data/cache_embeddings/finbert_title.npy')
emb_bert_t = np.load('data/cache_embeddings/bert_title.npy')

df_raw['Tanggal_dt'] = pd.to_datetime(df_raw['Tanggal_dt'])
df_raw = df_raw.sort_values('Tanggal_dt').reset_index(drop=True)
df_raw['rate_clean'] = (df_raw['Kurs Jual'] + df_raw['Kurs Beli']) / 2.0

# Precompile entity regex
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
        
        us_cnt = len(US_CASE.findall(full_t)) + len(US_LOWER.findall(full_t.lower()))
        idr_cnt = len(IDR_CASE.findall(full_t)) + len(IDR_LOWER.findall(full_t.lower()))
        w_us = 1.0 if (us_title or us_cnt >= 2) else 0.0
        w_idr = 1.0 if (idr_title or idr_cnt >= 1) else 0.0
        
        toks = full_t.lower().split()
        gpr_d = sum(1 for w in toks if w in GPR_SET) / max(len(toks), 1)
        
        art_feats.append({
            'pol_bin': pol_bin,
            'pol_margin': pol_margin,
            'flag_us': w_us,
            'flag_idr': w_idr,
            'entity_dir_bin': w_us * pol_bin - w_idr * 2.0 * pol_bin,
            'entity_dir_margin': w_us * pol_margin - w_idr * 2.0 * pol_margin,
            'gpr_dens': gpr_d,
            'gpr_weak_bin': gpr_d * (-pol_bin),
            'gpr_weak_margin': gpr_d * (-pol_margin),
        })
    if art_feats:
        agg = pd.DataFrame(art_feats).mean().to_dict()
        agg['has_news'] = 1.0
    else:
        agg = {k: 0.0 for k in ['pol_bin', 'pol_margin', 'flag_us', 'flag_idr',
                                'entity_dir_bin', 'entity_dir_margin', 'gpr_dens',
                                'gpr_weak_bin', 'gpr_weak_margin']}
        agg['has_news'] = 0.0
    agg['date'] = row['Tanggal_dt']
    agg['rate_clean'] = row['rate_clean']
    daily_rows.append(agg)

daily = pd.DataFrame(daily_rows).set_index('date').sort_index()

# Add daily FinBERT + BERT title embeddings
unm_dates = pd.to_datetime(unm['aligned_trading_date']).dt.normalize()
date_to_idx = {d: i for i, d in enumerate(daily.index)}
emb_comb = np.hstack([emb_bert_t, emb_fin_t])
X_emb_day = np.zeros((len(daily), emb_comb.shape[1]), dtype=np.float32)
for dt_val, grp in unm.groupby(unm_dates):
    if dt_val in date_to_idx:
        X_emb_day[date_to_idx[dt_val]] = emb_comb[grp.index].mean(axis=0)

# Targets: log_ret at t
daily['log_ret'] = np.log(daily['rate_clean'] / daily['rate_clean'].shift(1))
daily['target_dir'] = (daily['log_ret'] > 0).astype(int)

# Time-series basic features (strictly t-1)
daily['ret_lag1'] = daily['log_ret'].shift(1)
daily['ret_lag2'] = daily['log_ret'].shift(2)
daily['vol_5d']   = daily['log_ret'].shift(1).rolling(5).std()
consecs, cur = [], 0
for r in daily['log_ret']:
    if pd.isna(r): consecs.append(0)
    elif r > 0: cur = cur + 1 if cur > 0 else 1; consecs.append(cur)
    elif r < 0: cur = cur - 1 if cur < 0 else -1; consecs.append(cur)
    else: cur = 0; consecs.append(0)
daily['consec_up_days'] = pd.Series(consecs, index=daily.index).shift(1)
ma20 = daily['rate_clean'].shift(1).rolling(20).mean()
daily['dist_ma20'] = daily['rate_clean'].shift(1) / ma20 - 1.0

# Also create shifted(1) text features (yesterday's CNBC US-timezone news affecting today's Jakarta open/rate!)
base_txt_cols = ['has_news', 'pol_bin', 'pol_margin', 'flag_us', 'flag_idr',
                 'entity_dir_bin', 'entity_dir_margin', 'gpr_dens', 'gpr_weak_bin', 'gpr_weak_margin']
for c in base_txt_cols:
    daily[f'{c}_lag1'] = daily[c].shift(1)

# Emb PCA (fit on train) using lagged t-1 news AND current t news
valid_mask = daily[['log_ret', 'ret_lag2', 'vol_5d', 'consec_up_days', 'dist_ma20', 'pol_bin_lag1']].notna().all(axis=1).values
valid = daily[valid_mask].copy()
X_emb_v = X_emb_day[valid_mask]
X_emb_v_lag1 = np.vstack([np.zeros((1, X_emb_day.shape[1])), X_emb_day[:-1]])[valid_mask]

n = len(valid)
n_tr = int(n * 0.70)
n_va = int(n * 0.15)
tr_m = np.arange(n) < n_tr
va_m = (np.arange(n) >= n_tr) & (np.arange(n) < n_tr + n_va)
te_m = np.arange(n) >= n_tr + n_va

# Fit PCA on train embeddings (lag1 + current)
sc_emb0 = StandardScaler().fit(X_emb_v[tr_m])
pca0 = PCA(n_components=8, random_state=SEED).fit(sc_emb0.transform(X_emb_v[tr_m]))
Z0 = pca0.transform(sc_emb0.transform(X_emb_v))

sc_emb1 = StandardScaler().fit(X_emb_v_lag1[tr_m])
pca1 = PCA(n_components=8, random_state=SEED).fit(sc_emb1.transform(X_emb_v_lag1[tr_m]))
Z1 = pca1.transform(sc_emb1.transform(X_emb_v_lag1))

for k in range(8):
    valid[f'pca_t_{k}'] = Z0[:, k]
    valid[f'pca_lag1_{k}'] = Z1[:, k]

# Compare Stage 1 text classifiers
sets = {
    "A. Teks t (Entity+GPR+Pol)": base_txt_cols,
    "B. Teks t-1 (Entity+GPR+Pol lag1)": [f"{c}_lag1" for c in base_txt_cols],
    "C. Teks t & t-1 + PCA8(lag1)": base_txt_cols + [f"{c}_lag1" for c in base_txt_cols] + [f"pca_lag1_{k}" for k in range(8)],
    "D. Teks t-1 + PCA8(lag1) only": [f"{c}_lag1" for c in base_txt_cols] + [f"pca_lag1_{k}" for k in range(8)],
}

y_dir = valid['target_dir'].values
print("=== STAGE 1 TEXT DIRECTION CLASSIFIER COMPARISON ===")
best_probs = {}
for sname, cols in sets.items():
    sc = StandardScaler().fit(valid.loc[tr_m, cols])
    Xtr = sc.transform(valid.loc[tr_m, cols])
    Xva = sc.transform(valid.loc[va_m, cols])
    Xte = sc.transform(valid.loc[te_m, cols])
    
    for cname, clf in [
        ("LogReg", LogisticRegression(C=0.05, random_state=SEED)),
        ("GBDT", GradientBoostingClassifier(n_estimators=40, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=SEED)),
        ("MLP", MLPClassifier(hidden_layer_sizes=(16,), alpha=3.0, max_iter=300, random_state=SEED)),
    ]:
        clf.fit(Xtr, y_dir[tr_m])
        p_te = clf.predict_proba(Xte)[:, 1]
        pred = (p_te >= 0.5).astype(int)
        acc = accuracy_score(y_dir[te_m], pred) * 100
        bacc = balanced_accuracy_score(y_dir[te_m], pred) * 100
        auc = roc_auc_score(y_dir[te_m], p_te)
        cm = confusion_matrix(y_dir[te_m], pred).ravel()
        print(f"  {sname:32s} | {cname:6s}: Acc={acc:.2f}% BalAcc={bacc:.2f}% AUC={auc:.4f} CM={cm}")
        if sname.startswith("C.") and cname == "GBDT":
            best_probs['C_GBDT'] = (clf, sc, cols)

# Now test Stage 2 with OOF Stage 1 probability + Basic Time-Series!
print("\n=== STAGE 2 LOG_RETURN PREDICTION (OOF TEXT PROB + BASIC TS) ===")
cols_s1 = sets["C. Teks t & t-1 + PCA8(lag1)"]
sc1 = StandardScaler().fit(valid.loc[tr_m, cols_s1])
X1_tr = sc1.transform(valid.loc[tr_m, cols_s1])
X1_va = sc1.transform(valid.loc[va_m, cols_s1])
X1_te = sc1.transform(valid.loc[te_m, cols_s1])

kf = KFold(n_splits=5, shuffle=False)
oof_prob = np.zeros(tr_m.sum())
for f_tr, f_va in kf.split(X1_tr):
    c_f = GradientBoostingClassifier(n_estimators=40, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=SEED)
    c_f.fit(X1_tr[f_tr], y_dir[tr_m][f_tr])
    oof_prob[f_va] = c_f.predict_proba(X1_tr[f_va])[:, 1]

c_full = GradientBoostingClassifier(n_estimators=40, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=SEED).fit(X1_tr, y_dir[tr_m])
valid.loc[tr_m, 'prob_up_oof'] = oof_prob
valid.loc[va_m, 'prob_up_oof'] = c_full.predict_proba(X1_va)[:, 1]
valid.loc[te_m, 'prob_up_oof'] = c_full.predict_proba(X1_te)[:, 1]

ts_cols = ['ret_lag1', 'ret_lag2', 'vol_5d', 'consec_up_days', 'dist_ma20']
s2_cols = ts_cols + ['prob_up_oof', 'entity_dir_margin_lag1', 'gpr_weak_margin_lag1']

sc2 = StandardScaler().fit(valid.loc[tr_m, s2_cols])
X2_tr = sc2.transform(valid.loc[tr_m, s2_cols])
X2_va = sc2.transform(valid.loc[va_m, s2_cols])
X2_te = sc2.transform(valid.loc[te_m, s2_cols])

y_tr = valid.loc[tr_m, 'log_ret'].values
y_va = valid.loc[va_m, 'log_ret'].values
y_te = valid.loc[te_m, 'log_ret'].values
mu_y, std_y = y_tr.mean(), y_tr.std()
y_tr_s = (y_tr - mu_y) / std_y

def show_m(name, y_p):
    err = y_te - y_p
    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    mse = np.mean(err**2)
    da = np.mean(np.sign(y_te) == np.sign(y_p)) * 100
    bacc = balanced_accuracy_score((y_te > 0).astype(int), (y_p > 0).astype(int)) * 100
    print(f"  {name:28s}: RMSE={rmse:.6f} MAE={mae:.6f} MSE={mse:.8f} DirAcc={da:.2f}% BalAcc={bacc:.2f}%")

show_m("Baseline Mean", np.full_like(y_te, mu_y))
for alpha in [20.0, 50.0, 100.0]:
    r_mod = Ridge(alpha=alpha, random_state=SEED).fit(X2_tr, y_tr_s)
    show_m(f"Ridge (alpha={alpha})", r_mod.predict(X2_te) * std_y + mu_y)

mlp = MLPRegressor(hidden_layer_sizes=(16, 8), alpha=5.0, max_iter=300, random_state=SEED).fit(X2_tr, y_tr_s)
show_m("MLP (16,8 alpha=5)", mlp.predict(X2_te) * std_y + mu_y)

gb = GradientBoostingRegressor(n_estimators=40, max_depth=2, learning_rate=0.03, subsample=0.8, random_state=SEED).fit(X2_tr, y_tr_s)
show_m("GBDT Regressor", gb.predict(X2_te) * std_y + mu_y)

p_r50 = Ridge(alpha=50.0, random_state=SEED).fit(X2_tr, y_tr_s).predict(X2_te) * std_y + mu_y
p_mlp = mlp.predict(X2_te) * std_y + mu_y
show_m("Blend (50% Ridge + 50% MLP)", 0.5 * p_r50 + 0.5 * p_mlp)
