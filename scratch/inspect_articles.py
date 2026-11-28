import pandas as pd

df = pd.read_csv('data/processed/merged_data_clean.csv')
print('news_count summary:')
print(df['news_count'].describe())
print('\nSample all_title split by |:')
for i in range(5):
    parts = str(df['all_title'].iloc[i]).split(' | ')
    nc = df['news_count'].iloc[i]
    print(f'Row {i} (date={df["Tanggal_dt"].iloc[i]}, news_count={nc}), split parts={len(parts)}:')
    for p in parts[:4]:
        print('  -', p[:100])

df_unm = pd.read_csv('data/processed/news_processed_unmerged.csv', nrows=5)
print('\nnews_processed_unmerged cols:', df_unm.columns.tolist())
