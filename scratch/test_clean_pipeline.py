import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('data/processed/merged_data_clean.csv')
if 'Kurs Jual' in df.columns and 'Kurs Beli' in df.columns:
    df['rate'] = (df['Kurs Jual'] + df['Kurs Beli']) / 2.0
elif 'rate_clean' in df.columns:
    df['rate'] = df['rate_clean']

df['Tanggal_dt'] = pd.to_datetime(df['Tanggal_dt'])
df = df.sort_values('Tanggal_dt').reset_index(drop=True)

df['ret'] = df['rate'].pct_change()

# Clean Temporal Features (strictly up to t-1)
df['ret_lag1'] = df['ret'].shift(1)
df['ret_lag2'] = df['ret'].shift(2)
df['ret_lag3'] = df['ret'].shift(3)
df['vol_5d']   = df['ret'].shift(1).rolling(5).std()
df['vol_20d']  = df['ret'].shift(1).rolling(20).std()
df['mom_5d']   = df['rate'].shift(1) / df['rate'].shift(6) - 1.0
df['mom_20d']  = df['rate'].shift(1) / df['rate'].shift(21) - 1.0
df['dow']      = df['Tanggal_dt'].dt.dayofweek

sent_cache = 'data/processed/finbert_sentiment.csv'
if os.path.exists(sent_cache):
    s_df = pd.read_csv(sent_cache)
    for col in ['sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_polarity']:
        df[col] = s_df[col]
else:
    for col in ['sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_polarity']:
        df[col] = 0.0

valid = df.dropna(subset=['ret', 'vol_20d', 'mom_20d', 'ret_lag3']).reset_index(drop=True)
N = len(valid)
n_train = int(N * 0.70)
n_val = int(N * 0.15)

tr = valid.iloc[:n_train]
te = valid.iloc[n_train+n_val:]

y_tr = tr['ret'].values
y_te = te['ret'].values
mu_tr = y_tr.mean()
std_tr = y_tr.std()

temporal_cols = ['ret_lag1', 'ret_lag2', 'ret_lag3', 'vol_5d', 'vol_20d', 'mom_5d', 'mom_20d', 'dow']
text_cols = ['sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_polarity']

# 1. Temporal Only
sc_temp = StandardScaler().fit(tr[temporal_cols])
X_tr_t = sc_temp.transform(tr[temporal_cols])
X_te_t = sc_temp.transform(te[temporal_cols])

m_t = Ridge(alpha=100.0, random_state=42)
m_t.fit(X_tr_t, (y_tr - mu_tr)/std_tr)
p_t = m_t.predict(X_te_t) * std_tr + mu_tr

# 2. Temporal + Text
comb_cols = temporal_cols + text_cols
sc_comb = StandardScaler().fit(tr[comb_cols])
X_tr_c = sc_comb.transform(tr[comb_cols])
X_te_c = sc_comb.transform(te[comb_cols])

m_c = Ridge(alpha=100.0, random_state=42)
m_c.fit(X_tr_c, (y_tr - mu_tr)/std_tr)
p_c = m_c.predict(X_te_c) * std_tr + mu_tr

def get_stats(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    valid_s = (y_true != 0) & (y_pred != 0)
    da = np.mean(np.sign(y_true[valid_s]) == np.sign(y_pred[valid_s])) * 100
    r2 = 1.0 - np.sum((y_true - y_pred)**2) / np.sum((y_true - mu_tr)**2)
    return f"RMSE={rmse:.6f}, MAE={mae:.6f}, DirAcc={da:.2f}%, R2={r2:.5f}"

print("Temporal Clean Only  :", get_stats(y_te, p_t))
print("Temporal + Text Clean:", get_stats(y_te, p_c))
