import json

md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Analisis Kata Pembeda: Hari Kurs Naik vs Kurs Turun (Teks Mentah Setelah Stopword Removal)\n",
        "Berbeda dengan *Word Cloud* biasa yang hanya menampilkan kata yang paling sering muncul secara global (seperti *oil*, *china*, *russia*), analisis ini membandingkan **frekuensi kemunculan relatif (%)** dan **Log-Odds Ratio** tiap kata langsung dari **teks mentah setelah *stopword removal* saja** (`all_title` dan `all_full_text`, **tanpa *lemmatization***) antara **hari saat kurs USD/IDR Naik (Rupiah melemah)** vs **hari saat kurs USD/IDR Turun (Rupiah menguat)**.\n",
        "\n",
        "Kata dengan nilai $\\log_2\\text{Odds}$ positif tertinggi condong muncul pada hari kurs **NAIK**, sedangkan nilai negatif tertinggi condong muncul pada hari kurs **TURUN** — menunjukkan kata mana yang benar-benar memiliki **daya pembeda untuk klasifikasi arah**."
    ]
}

code_lines = r"""# Analisis & Visualisasi Kata Pembeda: Hari Kurs Naik vs Hari Kurs Turun (Teks Mentah + Stopword Removal)
import re
import html
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# 1. Gunakan merged_data_clean dari memori jika tersedia, atau muat dari CSV
if "merged_data_clean" in globals():
    df_disc = merged_data_clean.copy()
else:
    csv_cand = [
        Path("data/processed/merged_data_clean.csv"),
        Path("../data/processed/merged_data_clean.csv"),
        Path("percobaan task 3/data/processed/merged_data_clean.csv"),
    ]
    csv_p = next((p for p in csv_cand if p.exists()), None)
    if csv_p is None:
        raise FileNotFoundError("File merged_data_clean.csv tidak ditemukan!")
    df_disc = pd.read_csv(csv_p)

df_disc["Tanggal_dt"] = pd.to_datetime(df_disc["Tanggal_dt"])
df_disc = df_disc.sort_values("Tanggal_dt").reset_index(drop=True)
df_disc["rate_mid"] = (df_disc["Kurs Jual"] + df_disc["Kurs Beli"]) / 2.0
df_disc["ret"] = df_disc["rate_mid"].pct_change()

# Ambil hanya hari bursa yang memiliki berita dan pergerakan kurs (naik/turun)
df_disc = df_disc[(df_disc["news_count"] > 0) & df_disc["ret"].notna() & (df_disc["ret"] != 0)].copy()
df_disc["is_up"] = (df_disc["ret"] > 0).astype(int)  # 1 = Kurs Naik (IDR Melemah), 0 = Kurs Turun (IDR Menguat)

n_up_days = int((df_disc["is_up"] == 1).sum())
n_dn_days = int((df_disc["is_up"] == 0).sum())
print(f"Total hari berberita dianalisis: {len(df_disc)} hari (Naik={n_up_days} hari, Turun={n_dn_days} hari)")

# Gabungkan stopword Bahasa Inggris + filler editorial berita + pecahan kontraksi (tanpa lemmatization)
stop_base = set(STOPWORDS_EN) if "STOPWORDS_EN" in globals() else set(ENGLISH_STOP_WORDS)
STOP_DISC = stop_base | set(ENGLISH_STOP_WORDS) | {
    "cnbc", "said", "says", "say", "article", "read", "also", "would", "could",
    "new", "year", "years", "one", "two", "week", "weeks", "day", "days",
    "month", "months", "time", "percent", "today", "bad", "good", "must", "used",
    "don", "didn", "doesn", "isn", "wasn", "aren", "won", "wouldn", "couldn", "shouldn",
    "haven", "hasn", "hadn"
}

def compute_discriminative_words(df_in, col_name, min_days=10):
    up_cnt, dn_cnt = Counter(), Counter()
    for _, row in df_in.iterrows():
        raw = str(row[col_name])
        if not raw or raw.lower() == "nan":
            continue
        # Pembersihan teks mentah + stopword removal saja (tanpa stemming / lemmatization)
        t = html.unescape(raw.lower()).replace("’", "'").replace("‘", "'")
        t = re.sub(r"http\S+|www\.\S+|<[^>]+>", " ", t)
        t = re.sub(r"'s\b", "", t)
        t = re.sub(r"[^a-z\s]", " ", t)
        toks = [w for w in t.split() if len(w) > 2 and w not in STOP_DISC]

        # Hitung kemunculan unik per hari (Document Frequency) agar 1 artikel panjang tidak mendominasi
        unique_toks = set(toks)
        if row["is_up"] == 1:
            up_cnt.update(unique_toks)
        else:
            dn_cnt.update(unique_toks)

    rows = []
    vocab = set(up_cnt.keys()) | set(dn_cnt.keys())
    for w in vocab:
        cu, cd = up_cnt[w], dn_cnt[w]
        if cu + cd < min_days:
            continue
        # Proporsi kemunculan per kelas & Smoothed Log2-Odds Ratio
        p_up = (cu + 0.5) / (n_up_days + 1.0)
        p_dn = (cd + 0.5) / (n_dn_days + 1.0)
        log2_odds = float(np.log2((p_up / (1.0 - p_up)) / (p_dn / (1.0 - p_dn))))
        pct_up = cu / n_up_days * 100.0
        pct_dn = cd / n_dn_days * 100.0
        rows.append({
            "Kata": w,
            "Hari_Naik": cu,
            "Hari_Turun": cd,
            "Pct_Naik (%)": round(pct_up, 2),
            "Pct_Turun (%)": round(pct_dn, 2),
            "Selisih_Pct (%)": round(pct_up - pct_dn, 2),
            "Log2_Odds": round(log2_odds, 4)
        })
    return pd.DataFrame(rows)

# 2. Analisis pada Teks Mentah setelah Stopword Removal (all_title & all_full_text, tanpa lemmatization)
disc_title = compute_discriminative_words(df_disc, "all_title", min_days=10)
disc_body  = compute_discriminative_words(df_disc, "all_full_text", min_days=40)

print("\n=== TOP 10 KATA PEMBEDA DI JUDUL MENTAH (STOPWORD REMOVAL): CONDONG KURS NAIK (RUPIAH MELEMAH) ===")
display(disc_title.nlargest(10, "Log2_Odds").reset_index(drop=True))

print("\n=== TOP 10 KATA PEMBEDA DI JUDUL MENTAH (STOPWORD REMOVAL): CONDONG KURS TURUN (RUPIAH MENGUAT) ===")
display(disc_title.nsmallest(10, "Log2_Odds").reset_index(drop=True))

print("\n=== TOP 10 KATA PEMBEDA DI FULL-TEXT MENTAH (STOPWORD REMOVAL): CONDONG KURS NAIK ===")
display(disc_body.nlargest(10, "Log2_Odds").reset_index(drop=True))

print("\n=== TOP 10 KATA PEMBEDA DI FULL-TEXT MENTAH (STOPWORD REMOVAL): CONDONG KURS TURUN ===")
display(disc_body.nsmallest(10, "Log2_Odds").reset_index(drop=True))

# 3. Visualisasi Diverging Bar Chart (Judul Mentah vs Full-Text Mentah setelah Stopword Removal)
fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), dpi=150)

for ax, df_src, judul_panel in [
    (axes[0], disc_title, "Kata Pembeda pada Judul Mentah (all_title, Stopword Removal)"),
    (axes[1], disc_body,  "Kata Pembeda pada Full-Text Mentah (all_full_text, Stopword Removal)")
]:
    top_up = df_src.nlargest(10, "Log2_Odds")
    top_dn = df_src.nsmallest(10, "Log2_Odds")
    comb = pd.concat([top_dn, top_up]).drop_duplicates("Kata").sort_values("Log2_Odds")

    colors = ["#2a9d8f" if v < 0 else "#e63946" for v in comb["Log2_Odds"]]
    bars = ax.barh(comb["Kata"], comb["Log2_Odds"], color=colors, edgecolor="none", alpha=0.9)
    ax.axvline(0, color="black", linewidth=1.0)
    max_abs = float(comb["Log2_Odds"].abs().max())
    ax.set_xlim(-max_abs * 1.32, max_abs * 1.32)
    ax.set_title(f"{judul_panel}\n(Kiri/Hijau = Condong Kurs Turun | Kanan/Merah = Condong Kurs Naik)", fontweight="bold", fontsize=11)
    ax.set_xlabel("Log2 Odds Ratio (Daya Pembeda Arah Kurs)")
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    # Anotasi frekuensi hari (Naik vs Turun) di samping batang
    for bar, (_, r) in zip(bars, comb.iterrows()):
        val = r["Log2_Odds"]
        txt = f" {r['Hari_Naik']} vs {r['Hari_Turun']} "
        ha = "left" if val >= 0 else "right"
        ax.text(val, bar.get_y() + bar.get_height()/2, txt, va="center", ha=ha, fontsize=8, color="#333333")

plt.tight_layout()
fig_dir = Path("percobaan task 3/reports/figures")
if not fig_dir.parent.parent.exists():
    fig_dir = Path("reports/figures")
fig_dir.mkdir(parents=True, exist_ok=True)
out_fig = fig_dir / "discriminative_words_up_vs_down.png"
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.show()
print(f"Grafik kata pembeda berhasil disimpan ke: {out_fig}")
"""

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in code_lines.strip().split("\n")]
}

wc_md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Visualisasi Word Cloud Diskriminatif (Teks Mentah Setelah Stopword Removal)\n",
        "Menggunakan `generate_from_frequencies` berbobot **$|\\log_2\\text{Odds Ratio}|$** (Top 45 kata pembeda tiap arah dari **teks mentah setelah *stopword removal***), sehingga ukuran kata mencerminkan **daya pembeda** antara hari Kurs Naik vs Kurs Turun — bukan sekadar kata umum yang sering muncul di kedua kondisi."
    ]
}

wc_code_lines = r"""# Visualisasi Word Cloud Diskriminatif: Hari Kurs Naik vs Hari Kurs Turun (Teks Mentah + Stopword Removal)
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from wordcloud import WordCloud

# Pangkas ujung warna pucat agar seluruh kata kontras dan tajam di latar putih
cmap_up   = ListedColormap(plt.cm.Reds(np.linspace(0.45, 0.95, 256)))
cmap_down = ListedColormap(plt.cm.YlGn(np.linspace(0.50, 0.95, 256)))

fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=150)

configs = [
    (axes[0, 0], disc_title, "up",   cmap_up,   "Judul Mentah (Stopword Removal) — Condong Kurs NAIK (Rupiah Melemah)"),
    (axes[0, 1], disc_title, "down", cmap_down, "Judul Mentah (Stopword Removal) — Condong Kurs TURUN (Rupiah Menguat)"),
    (axes[1, 0], disc_body,  "up",   cmap_up,   "Full-Text Mentah (Stopword Removal) — Condong Kurs NAIK (Rupiah Melemah)"),
    (axes[1, 1], disc_body,  "down", cmap_down, "Full-Text Mentah (Stopword Removal) — Condong Kurs TURUN (Rupiah Menguat)"),
]

for ax, df_src, direction, cmap, title in configs:
    if direction == "up":
        sub = df_src[df_src["Log2_Odds"] > 0].nlargest(45, "Log2_Odds")
        freqs = dict(zip(sub["Kata"], sub["Log2_Odds"]))
    else:
        sub = df_src[df_src["Log2_Odds"] < 0].nsmallest(45, "Log2_Odds")
        freqs = dict(zip(sub["Kata"], sub["Log2_Odds"].abs()))

    wc = WordCloud(
        width=800,
        height=450,
        background_color="white",
        colormap=cmap,
        max_words=45,
        prefer_horizontal=0.85,
        random_state=42
    ).generate_from_frequencies(freqs)

    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(title, fontweight="bold", fontsize=11, pad=8)
    ax.axis("off")

plt.tight_layout()
fig_dir = Path("percobaan task 3/reports/figures")
if not fig_dir.parent.parent.exists():
    fig_dir = Path("reports/figures")
fig_dir.mkdir(parents=True, exist_ok=True)
out_wc = fig_dir / "wordcloud_discriminative_up_vs_down.png"
plt.savefig(out_wc, dpi=150, bbox_inches="tight")
plt.show()
print(f"Word Cloud diskriminatif berhasil disimpan ke: {out_wc}")
"""

wc_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in wc_code_lines.strip().split("\n")]
}

nb_path = "percobaan task 3/preprocessing.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Hindari duplikasi sel apabila skrip dijalankan lebih dari sekali
titles = ["".join(c.get("source", [])) for c in nb["cells"]]
if any("Analisis Kata Pembeda: Hari Kurs Naik vs Kurs Turun" in t for t in titles):
    idx_md = next(i for i, t in enumerate(titles) if "Analisis Kata Pembeda: Hari Kurs Naik vs Kurs Turun" in t)
    nb["cells"] = nb["cells"][:idx_md]

nb["cells"].extend([md_cell, code_cell, wc_md_cell, wc_code_cell])

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Berhasil memperbarui {nb_path} (Total sel sekarang: {len(nb['cells'])})")
