import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/merged_data_clean.csv')
print("Shape:", df.shape)

# Hitung kurs tengah
if 'Kurs Jual' in df.columns and 'Kurs Beli' in df.columns:
    df['rate'] = (df['Kurs Jual'] + df['Kurs Beli']) / 2.0
elif 'rate_clean' in df.columns:
    df['rate'] = df['rate_clean']

# Sort by Tanggal
df['Tanggal_dt'] = pd.to_datetime(df['Tanggal_dt'])
df = df.sort_values('Tanggal_dt').reset_index(drop=True)

# Hitung return harian
df['ret'] = df['rate'].pct_change()
df = df.dropna(subset=['ret']).reset_index(drop=True)

N = len(df)
n_train = int(N * 0.70)
n_val = int(N * 0.15)
n_test = N - n_train - n_val

y_train = df.iloc[:n_train]['ret'].values
y_val = df.iloc[n_train:n_train+n_val]['ret'].values
y_test = df.iloc[n_train+n_val:]['ret'].values

print(f"Data counts: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
print(f"y_test std: {np.std(y_test):.6f}")
print(f"y_test mean: {np.mean(y_test):.6f}")

# Naive 0 prediction RMSE:
rmse_zero = np.sqrt(np.mean(y_test**2))
mae_zero = np.mean(np.abs(y_test))
print(f"Naive Zero: RMSE = {rmse_zero:.6f}, MAE = {mae_zero:.6f}")

# Mean train prediction RMSE:
mu_tr = np.mean(y_train)
rmse_mean = np.sqrt(np.mean((y_test - mu_tr)**2))
mae_mean = np.mean(np.abs(y_test - mu_tr))
print(f"Mean Train: RMSE = {rmse_mean:.6f}, MAE = {mae_mean:.6f}")
