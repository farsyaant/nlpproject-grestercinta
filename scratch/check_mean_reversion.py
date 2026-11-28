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

# Check technical indicators: RSI, Mean Reversion
# Consecutive up days
daily['consec_up'] = 0
cur = 0
consecs = []
for r in daily['ret']:
    if pd.isna(r):
        consecs.append(0)
    elif r > 0:
        cur = cur + 1 if cur > 0 else 1
        consecs.append(cur)
    elif r < 0:
        cur = cur - 1 if cur < 0 else -1
        consecs.append(cur)
    else:
        cur = 0
        consecs.append(0)
daily['consec_sign'] = consecs
daily['ret_lag1'] = daily['ret'].shift(1)
daily['ret_lag2'] = daily['ret'].shift(2)

# If yesterday was positive, what is prob today is positive?
sub = daily.dropna()
p_up_after_up = (sub[sub['ret_lag1'] > 0]['ret'] > 0).mean()
p_up_after_down = (sub[sub['ret_lag1'] < 0]['ret'] > 0).mean()

print(f"Probabilitas Hari Ini Naik jika Kemarin Naik  : {p_up_after_up*100:.2f}%")
print(f"Probabilitas Hari Ini Naik jika Kemarin Turun : {p_up_after_down*100:.2f}%")

# If up 2 days in a row:
p_up_after_2up = (sub[sub['consec_sign'].shift(1) >= 2]['ret'] > 0).mean()
p_up_after_2down = (sub[sub['consec_sign'].shift(1) <= -2]['ret'] > 0).mean()
print(f"Probabilitas Naik jika sudah 2 hari berturut Naik : {p_up_after_2up*100:.2f}%")
print(f"Probabilitas Naik jika sudah 2 hari berturut Turun: {p_up_after_2down*100:.2f}%")
