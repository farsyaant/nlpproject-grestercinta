import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/merged_data_clean.csv')
date_col = 'Tanggal_dt' if 'Tanggal_dt' in df.columns else 'tanggal'
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col).reset_index(drop=True)

if 'Kurs Jual' in df.columns and 'Kurs Beli' in df.columns:
    df['rate'] = (df['Kurs Jual'] + df['Kurs Beli']) / 2.0
elif 'rate_clean' in df.columns:
    df['rate'] = df['rate_clean']

daily = df.groupby(date_col).agg({'rate': 'first'})
daily['ret'] = daily['rate'].pct_change()
daily['ret_up'] = (daily['ret'] > 0).astype(int)

# Check distribution in Train, Val, Test
n = len(daily.dropna())
n_tr = int(n * 0.7)
n_va = int(n * 0.15)
train = daily.iloc[1:n_tr+1]
val   = daily.iloc[n_tr+1:n_tr+1+n_va]
test  = daily.iloc[n_tr+1+n_va:]

print("=== DISTRIBUSI ARAH NAIK / TURUN (RET > 0 vs RET < 0) ===")
for name, d in [('Train', train), ('Val', val), ('Test', test), ('Full', daily.dropna())]:
    up = (d['ret'] > 0).sum()
    down = (d['ret'] < 0).sum()
    flat = (d['ret'] == 0).sum()
    total = len(d)
    print(f"{name:5s}: Total={total:4d} | Naik(USD up/IDR down)={up:3d} ({up/total*100:5.1f}%) | Turun={down:3d} ({down/total*100:5.1f}%) | Tetap={flat:2d}")

# Check autocorrelation of returns
print("\n=== AUTOKORELASI RETURN KURS (LAG 1 s/d 5) ===")
for lag in range(1, 6):
    corr = daily['ret'].corr(daily['ret'].shift(lag))
    print(f"Lag {lag}: Autokorelasi = {corr:+.4f}")
