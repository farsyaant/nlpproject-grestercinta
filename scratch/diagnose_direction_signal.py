import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# 1. Load Data
df = pd.read_csv('data/processed/merged_data_clean.csv')
date_col = 'Tanggal_dt' if 'Tanggal_dt' in df.columns else 'tanggal'
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col).reset_index(drop=True)

if 'Kurs Jual' in df.columns and 'Kurs Beli' in df.columns:
    df['rate'] = (df['Kurs Jual'] + df['Kurs Beli']) / 2.0
elif 'rate_clean' in df.columns:
    df['rate'] = df['rate_clean']

# Sentimen FinBERT
sent_cache = 'data/processed/finbert_sentiment.csv'
if os.path.exists(sent_cache):
    s_df = pd.read_csv(sent_cache)
    for col in ['sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_polarity']:
        df[col] = s_df[col].values

# BERTopic
topic_cache = 'data/processed/bertopic_probabilities.csv'
if os.path.exists(topic_cache):
    t_df = pd.read_csv(topic_cache)
    for c in t_df.columns:
        df[c] = t_df[c].values

# GPR
GPR_TERMS = ["war", "military", "conflict", "geopolitic", "threat", "sanction", "crisis", "attack", "invasion", "nuclear", "terrorism", "perang", "militer", "konflik", "geopolitik", "krisis", "sanksi"]
def extract_gpr(t, b):
    comb = (str(t) + " " + str(b)).lower()
    toks = comb.split()
    if not toks: return 0, 0.0
    c = sum(1 for tok in toks if any(g in tok for g in GPR_TERMS))
    return c, c / len(toks)

gpr_res = [extract_gpr(t, b) for t, b in zip(df['all_title'], df['all_full_text'])]
df['gpr_count'] = [r[0] for r in gpr_res]
df['gpr_density'] = [r[1] for r in gpr_res]

# Aggregasi Harian
agg_dict = {
    'rate': 'first',
    'gpr_count': 'mean',
    'gpr_density': 'mean',
    'sentiment_pos': 'mean',
    'sentiment_neg': 'mean',
    'sentiment_neu': 'mean',
    'sentiment_polarity': 'mean',
}
for c in df.columns:
    if c.startswith('topic_prob_'):
        agg_dict[c] = 'mean'

daily = df.groupby(date_col).agg(agg_dict)
daily['ret'] = daily['rate'].pct_change()
daily['ret_up'] = (daily['ret'] > 0).astype(int)
daily['ret_magnitude'] = daily['ret'].abs()

# Cek tanggal & return
valid = daily.dropna(subset=['ret']).copy()
n_total = len(valid)
n_up = (valid['ret'] > 0).sum()
n_down = (valid['ret'] < 0).sum()
n_flat = (valid['ret'] == 0).sum()

print(f"Total Hari Bursa: {n_total}")
print(f"Distribusi Return: UP={n_up} ({n_up/n_total*100:.1f}%), DOWN={n_down} ({n_down/n_total*100:.1f}%), FLAT={n_flat}")
print(f"Mean Return: {valid['ret'].mean():.6f}, Std Return: {valid['ret'].std():.6f}")

print("\n--- KORELASI FITUR TEKS DENGAN RETURN HARI INI (ret_t) ---")
for col in ['sentiment_polarity', 'sentiment_pos', 'sentiment_neg', 'gpr_count', 'gpr_density'] + [c for c in valid.columns if c.startswith('topic_prob_')]:
    r, p = pearsonr(valid[col], valid['ret'])
    sr, sp = spearmanr(valid[col], valid['ret'])
    print(f"{col:20s}: Pearson r={r:+.4f} (p={p:.4f}), Spearman rho={sr:+.4f} (p={sp:.4f})")

print("\n--- KORELASI FITUR TEKS DENGAN RETURN HARI BERIKUTNYA (ret_{t+1}) ---")
valid['ret_next'] = valid['ret'].shift(-1)
valid_next = valid.dropna(subset=['ret_next'])
for col in ['sentiment_polarity', 'sentiment_pos', 'sentiment_neg', 'gpr_count', 'gpr_density']:
    r, p = pearsonr(valid_next[col], valid_next['ret_next'])
    print(f"{col:20s}: Pearson r={r:+.4f} (p={p:.4f})")

print("\n--- KORELASI FITUR TEKS DENGAN MAGNITUDE (|ret_t|) ---")
for col in ['sentiment_polarity', 'sentiment_pos', 'sentiment_neg', 'gpr_count', 'gpr_density']:
    r, p = pearsonr(valid[col], valid['ret_magnitude'])
    print(f"{col:20s}: Pearson r={r:+.4f} (p={p:.4f})")

# Periksa Sampel Berita vs Sentimen vs Return
print("\n--- SAMPLE 5 HARI RETURN EKSTRIM NAIK (USD MENGUAT / RUPIAH MELEMAH PALING BANYAK) ---")
top_up = valid.nlargest(5, 'ret')
for dt, row in top_up.iterrows():
    print(f"Date: {dt.date()}, ret: {row['ret']:+.4f}, sent_pol: {row['sentiment_polarity']:+.4f}, gpr: {row['gpr_count']}")

print("\n--- SAMPLE 5 HARI RETURN EKSTRIM TURUN (RUPIAH MENGUAT PALING BANYAK) ---")
top_down = valid.nsmallest(5, 'ret')
for dt, row in top_down.iterrows():
    print(f"Date: {dt.date()}, ret: {row['ret']:+.4f}, sent_pol: {row['sentiment_polarity']:+.4f}, gpr: {row['gpr_count']}")
