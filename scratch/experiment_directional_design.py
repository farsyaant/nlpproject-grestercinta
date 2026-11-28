import os
import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
import torch
import torch.nn as nn

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1. Load datasets
df_raw = pd.read_csv('data/processed/merged_data_clean.csv')
fin_df = pd.read_csv('data/processed/finbert_unique_titles.csv')
fin_map = fin_df.set_index('title')[['finbert_pos', 'finbert_neg', 'finbert_neu']].to_dict('index')

date_col = 'Tanggal_dt'
df_raw[date_col] = pd.to_datetime(df_raw[date_col])
df_raw = df_raw.sort_values(date_col).reset_index(drop=True)
df_raw['rate_clean'] = (df_raw['Kurs Jual'] + df_raw['Kurs Beli']) / 2.0

# 2. Entity & GPR Patterns (Careful: NO lowercase "us" pronoun match!)
US_CASE = r"\b(U\.S\.?|USA|US|USD|Fed|FOMC|Powell|Yellen|Bessent|Treasury|Wall Street|White House|Trump|Biden)\b"
US_LOWER = r"\b(united states|america|american|federal reserve|the fed|dollar|greenback)\b"

IDR_CASE = r"\b(IDR|BI|IHSG|Rupiah|Indonesia|Indonesian|Jakarta|Jokowi|Prabowo|Sri Mulyani|Perry Warjiyo|Bank Indonesia)\b"
IDR_LOWER = r"\b(indonesia|indonesian|rupiah|jakarta|bank indonesia|bi rate|bi-rate|ihsg)\b"

GPR_TERMS = [
    "war", "military", "conflict", "geopolitic", "geopolitical", "threat", "sanction", "sanctions",
    "tension", "tensions", "crisis", "attack", "invasion", "nuclear", "terrorism", "terrorist",
    "missile", "hostility", "clash", "rebel", "insurgency", "troops", "sovereignty", "escalation",
    "retaliation", "dispute", "standoff", "blockade", "weapon", "drills", "border", "combat",
    "tariff", "tariffs", "oil", "iran", "russia", "ukraine", "israel", "gaza", "taiwan"
]

# 3. Article-level feature extraction & daily aggregation
daily_rows = []
for idx, row in df_raw.iterrows():
    titles = str(row['all_title']).split(' | ') if pd.notna(row['all_title']) else []
    bodies = str(row['all_full_text']).split(' | ') if pd.notna(row['all_full_text']) else []
    
    art_feats = []
    for i, t in enumerate(titles):
        t_c = t.strip()
        if not t_c or t_c.lower() == 'nan':
            continue
        b_c = bodies[i].strip() if i < len(bodies) else ""
        if b_c.lower() == 'nan':
            b_c = ""
        
        # FinBERT sentiment (pure positive vs negative)
        scores = fin_map.get(t_c, {'finbert_pos': 0.5, 'finbert_neg': 0.5, 'finbert_neu': 0.0})
        pos, neg = scores['finbert_pos'], scores['finbert_neg']
        # Bipolar score in [-1, 1] excluding neutral dilution, AND strict sign {-1, +1}
        pol_margin = (pos - neg) / (pos + neg + 1e-8)
        pol_binary = 1.0 if pos >= neg else -1.0
        
        # Entity flags: focus on Title + Lead (first 400 chars) + full text density
        lead_text = f"{t_c} {b_c[:400]}"
        full_text = f"{t_c} {b_c}"
        
        us_title = int(bool(re.search(US_CASE, t_c)) or bool(re.search(US_LOWER, t_c.lower())))
        idr_title = int(bool(re.search(IDR_CASE, t_c)) or bool(re.search(IDR_LOWER, t_c.lower())))
        
        us_cnt = len(re.findall(US_CASE, full_text)) + len(re.findall(US_LOWER, full_text.lower()))
        idr_cnt = len(re.findall(IDR_CASE, full_text)) + len(re.findall(IDR_LOWER, full_text.lower()))
        
        # Weighted entity flags (if mentioned in title weight 2.0, else proportional to mentions)
        w_us = 1.0 if (us_title or us_cnt >= 2) else 0.0
        w_idr = 1.0 if (idr_title or idr_cnt >= 1) else 0.0
        
        # Entity-Weighted Target Sentiment:
        # US Pos -> USD strengthens -> USD/IDR UP (+1)
        # IDR Pos -> IDR strengthens -> USD/IDR DOWN (-1)
        # Apply extra boost when Indonesia specifically mentioned (rare & direct)
        entity_dir_bin = (w_us * pol_binary) - (w_idr * 2.0 * pol_binary)
        entity_dir_margin = (w_us * pol_margin) - (w_idr * 2.0 * pol_margin)
        
        # GPR count & density
        toks = full_text.lower().split()
        n_tok = len(toks) if toks else 1
        gpr_c = sum(1 for tok in toks if any(g in tok for g in GPR_TERMS))
        gpr_dens = gpr_c / n_tok
        
        # GPR + Polarity interaction:
        # High GPR + Negative polarity -> Rupiah weakens (+1)
        # High GPR + Positive polarity -> tension eases / Rupiah strengthens (-1)
        gpr_weak_bin = gpr_dens * (-pol_binary)   # +gpr_dens if pol < 0, -gpr_dens if pol > 0
        gpr_weak_margin = gpr_dens * (-pol_margin)
        
        art_feats.append({
            'has_news': 1.0,
            'pol_binary': pol_binary,
            'pol_margin': pol_margin,
            'flag_us': w_us,
            'flag_idr': w_idr,
            'us_title': float(us_title),
            'idr_title': float(idr_title),
            'us_sent': w_us * pol_binary,
            'idr_sent': w_idr * pol_binary,
            'entity_dir_bin': entity_dir_bin,
            'entity_dir_margin': entity_dir_margin,
            'gpr_dens': gpr_dens,
            'gpr_weak_bin': gpr_weak_bin,
            'gpr_weak_margin': gpr_weak_margin,
        })
    
    if art_feats:
        agg = pd.DataFrame(art_feats).mean().to_dict()
        agg['news_count'] = len(art_feats)
    else:
        agg = {
            'has_news': 0.0, 'pol_binary': 0.0, 'pol_margin': 0.0,
            'flag_us': 0.0, 'flag_idr': 0.0, 'us_title': 0.0, 'idr_title': 0.0,
            'us_sent': 0.0, 'idr_sent': 0.0, 'entity_dir_bin': 0.0, 'entity_dir_margin': 0.0,
            'gpr_dens': 0.0, 'gpr_weak_bin': 0.0, 'gpr_weak_margin': 0.0, 'news_count': 0.0
        }
    agg['date'] = row[date_col]
    agg['rate_clean'] = row['rate_clean']
    daily_rows.append(agg)

daily = pd.DataFrame(daily_rows).set_index('date').sort_index()
daily['log_ret'] = np.log(daily['rate_clean'] / daily['rate_clean'].shift(1))
daily['target_dir'] = (daily['log_ret'] > 0).astype(int)

# Basic time-series features (strictly lagged t-1)
daily['ret_lag1'] = daily['log_ret'].shift(1)
daily['ret_lag2'] = daily['log_ret'].shift(2)
daily['vol_5d']   = daily['log_ret'].shift(1).rolling(5).std()

consecs = []
cur = 0
for r in daily['log_ret']:
    if pd.isna(r): consecs.append(0)
    elif r > 0: cur = cur + 1 if cur > 0 else 1; consecs.append(cur)
    elif r < 0: cur = cur - 1 if cur < 0 else -1; consecs.append(cur)
    else: cur = 0; consecs.append(0)
daily['consec_up_days'] = pd.Series(consecs, index=daily.index).shift(1)
ma20 = daily['rate_clean'].shift(1).rolling(20).mean()
daily['dist_ma20'] = daily['rate_clean'].shift(1) / ma20 - 1.0

# Also lagged text features? Let's test current aligned (pre-15:00) vs 3-day rolling sentiment
daily['pol_margin_roll3'] = daily['pol_margin'].rolling(3, min_periods=1).mean()
daily['gpr_weak_roll3'] = daily['gpr_weak_margin'].rolling(3, min_periods=1).mean()

valid = daily.dropna(subset=['log_ret', 'ret_lag2', 'vol_5d', 'consec_up_days', 'dist_ma20']).copy()
n = len(valid)
n_tr = int(n * 0.70)
n_va = int(n * 0.15)
tr_m = np.arange(n) < n_tr
va_m = (np.arange(n) >= n_tr) & (np.arange(n) < n_tr + n_va)
te_m = np.arange(n) >= n_tr + n_va

print("=== KORELASI FITUR DENGAN LOG_RET (TRAIN SET SAJA) ===")
tr_df = valid.iloc[tr_m]
check_cols = ['pol_binary', 'pol_margin', 'flag_us', 'flag_idr', 'us_sent', 'idr_sent',
              'entity_dir_bin', 'entity_dir_margin', 'gpr_dens', 'gpr_weak_bin', 'gpr_weak_margin',
              'pol_margin_roll3', 'gpr_weak_roll3', 'ret_lag1', 'ret_lag2', 'vol_5d', 'consec_up_days', 'dist_ma20']
for c in check_cols:
    r, p = pearsonr(tr_df[c], tr_df['log_ret'])
    print(f"  {c:20s}: r={r:+.4f} (p={p:.4f})")

# Stage 1 Classification Tests
txt_cols = ['has_news', 'pol_binary', 'pol_margin', 'flag_us', 'flag_idr', 'us_sent', 'idr_sent',
            'entity_dir_bin', 'entity_dir_margin', 'gpr_dens', 'gpr_weak_bin', 'gpr_weak_margin']

sc1 = StandardScaler().fit(valid.iloc[tr_m][txt_cols])
X1_tr = sc1.transform(valid.iloc[tr_m][txt_cols])
X1_va = sc1.transform(valid.iloc[va_m][txt_cols])
X1_te = sc1.transform(valid.iloc[te_m][txt_cols])
y_dir_tr = valid.iloc[tr_m]['target_dir'].values
y_dir_va = valid.iloc[va_m]['target_dir'].values
y_dir_te = valid.iloc[te_m]['target_dir'].values

print(f"\nTest Set Class Balance: UP(1)={y_dir_te.mean()*100:.1f}%, DOWN(0)={(1-y_dir_te.mean())*100:.1f}%")

for name, clf in [
    ("LogReg (default)", LogisticRegression(C=0.1, random_state=SEED)),
    ("LogReg (balanced)", LogisticRegression(C=0.1, class_weight='balanced', random_state=SEED)),
    ("MLPClassifier (16,)", MLPClassifier(hidden_layer_sizes=(16,), alpha=2.0, max_iter=300, random_state=SEED)),
    ("GBDT Classifier", GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=SEED)),
]:
    clf.fit(X1_tr, y_dir_tr)
    p_te = clf.predict_proba(X1_te)[:, 1]
    pred = (p_te >= 0.5).astype(int)
    acc = accuracy_score(y_dir_te, pred) * 100
    bacc = balanced_accuracy_score(y_dir_te, pred) * 100
    auc = roc_auc_score(y_dir_te, p_te)
    cm = confusion_matrix(y_dir_te, pred).ravel()
    print(f"Stage 1 {name:20s}: Acc={acc:.2f}%, BalAcc={bacc:.2f}%, AUC={auc:.4f}, CM(TN,FP,FN,TP)={cm}")
