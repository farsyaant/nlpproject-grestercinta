import re
import numpy as np
import pandas as pd
from collections import Counter

df = pd.read_csv("data/processed/merged_data_clean.csv",
                 usecols=["Tanggal_dt", "Kurs Jual", "Kurs Beli", "news_count", "all_title", "all_text_lemmatized"])
df["Tanggal_dt"] = pd.to_datetime(df["Tanggal_dt"])
df = df.sort_values("Tanggal_dt").reset_index(drop=True)
df["rate_mid"] = (df["Kurs Jual"] + df["Kurs Beli"]) / 2.0
df["ret"] = df["rate_mid"].pct_change()
df = df[(df["news_count"] > 0) & df["ret"].notna() & (df["ret"] != 0)].copy()

df["is_up"] = (df["ret"] > 0).astype(int) # 1 = Kurs USD/IDR Naik (Rupiah Melemah), 0 = Turun (Rupiah Menguat)
print(f"Hari berita valid: {len(df)} | Hari Naik: {(df['is_up']==1).sum()} | Hari Turun: {(df['is_up']==0).sum()}")

# Hitung document frequency / word frequency di Title & Lemmatized Text
STOP_EXTRA = {"said", "says", "say", "cnbc", "also", "would", "could", "new", "year", "one", "two", "week", "day", "percent"}

def compute_word_discriminative_stats(df_in, col_name, min_total_freq=15, is_title=False):
    up_cnt, dn_cnt = Counter(), Counter()
    for _, row in df_in.iterrows():
        raw = str(row[col_name])
        if not raw or raw.lower() == "nan":
            continue
        if is_title:
            toks = re.findall(r"[a-z]{3,}", raw.lower())
        else:
            toks = [w for w in raw.split() if len(w) > 2 and w.isalpha()]
        toks = set(t for t in toks if t not in STOP_EXTRA) # Document frequency per hari (anti-spam artikel panjang)
        if row["is_up"] == 1:
            up_cnt.update(toks)
        else:
            dn_cnt.update(toks)
            
    n_up_days = (df_in["is_up"] == 1).sum()
    n_dn_days = (df_in["is_up"] == 0).sum()
    
    rows = []
    vocab = set(up_cnt.keys()) | set(dn_cnt.keys())
    for w in vocab:
        cu, cd = up_cnt[w], dn_cnt[w]
        if cu + cd < min_total_freq:
            continue
        p_up = (cu + 0.5) / (n_up_days + 1.0)
        p_dn = (cd + 0.5) / (n_dn_days + 1.0)
        log2_odds = np.log2((p_up / (1 - p_up)) / (p_dn / (1 - p_dn)))
        diff_pct = (cu / n_up_days - cd / n_dn_days) * 100
        rows.append({"word": w, "freq_naik": cu, "freq_turun": cd, "pct_naik": cu/n_up_days*100, "pct_turun": cd/n_dn_days*100, "diff_pct": diff_pct, "log2_odds": log2_odds})
    return pd.DataFrame(rows)

res_title = compute_word_discriminative_stats(df, "all_title", min_total_freq=10, is_title=True)
print("\n=== TOP 10 KATA PEMBEDA DI JUDUL: LEBIH SERING SAAT KURS NAIK (RUPIAH MELEMAH) ===")
print(res_title.nlargest(10, "log2_odds")[["word", "freq_naik", "freq_turun", "pct_naik", "pct_turun", "log2_odds"]].to_string(index=False))

print("\n=== TOP 10 KATA PEMBEDA DI JUDUL: LEBIH SERING SAAT KURS TURUN (RUPIAH MENGUAT) ===")
print(res_title.nsmallest(10, "log2_odds")[["word", "freq_naik", "freq_turun", "pct_naik", "pct_turun", "log2_odds"]].to_string(index=False))

res_lem = compute_word_discriminative_stats(df, "all_text_lemmatized", min_total_freq=40, is_title=False)
print("\n=== TOP 10 KATA PEMBEDA DI FULL-TEXT (LEMMATIZED): KURS NAIK ===")
print(res_lem.nlargest(10, "log2_odds")[["word", "freq_naik", "freq_turun", "pct_naik", "pct_turun", "log2_odds"]].to_string(index=False))

print("\n=== TOP 10 KATA PEMBEDA DI FULL-TEXT (LEMMATIZED): KURS TURUN ===")
print(res_lem.nsmallest(10, "log2_odds")[["word", "freq_naik", "freq_turun", "pct_naik", "pct_turun", "log2_odds"]].to_string(index=False))
