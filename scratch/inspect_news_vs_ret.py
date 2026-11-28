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

daily = df.groupby(date_col).agg({'rate': 'first', 'all_title': 'first'})
daily['ret'] = daily['rate'].pct_change()

print("--- 5 CONTOH BERITA KETIKA USD/IDR NAIK PALING TAJAM (RUPIAH MELEMAH / RET POSITIF BESAR) ---")
top_up = daily.nlargest(5, 'ret')
for dt, row in top_up.iterrows():
    print(f"\nTanggal: {dt.date()} | Return: {row['ret']:+.4%}")
    print(f"Judul: {str(row['all_title'])[:250]}")

print("\n--- 5 CONTOH BERITA KETIKA USD/IDR TURUN PALING TAJAM (RUPIAH MENGUAT / RET NEGATIF BESAR) ---")
top_down = daily.nsmallest(5, 'ret')
for dt, row in top_down.iterrows():
    print(f"\nTanggal: {dt.date()} | Return: {row['ret']:+.4%}")
    print(f"Judul: {str(row['all_title'])[:250]}")
