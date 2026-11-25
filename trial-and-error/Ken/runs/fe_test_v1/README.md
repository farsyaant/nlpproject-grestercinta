# FE test v1 — rekayasa fitur kurs + berita

Eksperimen kecil untuk menguji apakah fitur turunan berita menambah nilai di atas
Model A (7 fitur kurs) dan di atas naive 0. **Test set tidak dipakai** untuk memilih
fitur atau kombinasi.

```
python fe_pipeline.py --stage features     # fitur + uji anti-leakage
python fe_pipeline.py --stage screen       # 274 kombinasi (grup + acak)
python fe_pipeline.py --stage wide         # katalog besar (~1.7k kombinasi)
python fe_pipeline.py --stage select_wide  # kolom final
```

Sumber: `data/export/prices_daily.parquet` (1.194 baris) + `data/export/sentiment_daily.parquet`,
split dari kolom `split` (train 836 / val 178 / test 180), target `target` = `ret[t+1]`.
Metrik dari `src/eval_utils.py`.

## 1. Daftar fitur, formula, lookback

Semua fitur hari `t` hanya memakai data `<= t`. `n` = `news_count`,
`v` = `vader_title_mean`, `l` = `lm_text_net`, `u` = `lm_text_uncertainty`,
`vn` = `vader_title_neg_ratio`, `r` = `ret`, `s5`/`s20` = std return window 5/20.

| Grup | Fitur | Formula | Lookback |
|---|---|---|---|
| a_news | `fe_news_count` | `n[t]` | 0 |
| a_news | `fe_no_news` | `1[n[t]=0]` | 0 |
| a_news | `fe_news_ma5` | `mean(n[t-4..t])` | 5 |
| a_news | `fe_news_z20` | `(n[t]-mean(n[t-20..t-1]))/(std(...)+1)` | 21 |
| b_momentum | `fe_vader_d1` | `0 if n[t]=0 else v[t]-v_ffill[t-1]` | 1 |
| b_momentum | `fe_vader_d3` | `mean(v[t-2..t])-mean(v[t-5..t-3])` (observed-only) | 5 |
| b_momentum | `fe_lmnet_d1` | `0 if n[t]=0 else l[t]-l_ffill[t-1]` | 1 |
| b_momentum | `fe_vader_dailystd5` | `std(v_obs[t-4..t])`, min 2 obs, else 0 | 5 |
| b_momentum | `fe_vader_range5d` | `max-min(v_obs[t-4..t])` | 5 |
| b_momentum | `fe_title_text_gap` | `lm_title_net[t]-lm_text_net[t]` | 0 |
| c_decay | `fe_vader_ewm3` | `EWM(v[t], halflife=3, adjust=False)` | ~9 hari efektif |
| c_decay | `fe_vader_ewm10` | `EWM(v[t], halflife=10, adjust=False)` | ~30 hari efektif |
| c_decay | `fe_lmnet_ewm5` | `EWM(l[t], halflife=5, adjust=False)` | ~15 hari efektif |
| d_surprise | `fe_vader_surprise20` | `0 if n[t]=0 else (v[t]-mean(v[t-20..t-1]))/(std+0.01)` | 21 |
| d_surprise | `fe_lmnet_surprise20` | idem untuk `l` | 21 |
| d_surprise | `fe_lmunc_surprise20` | idem untuk `u` | 21 |
| e_interaction | `fe_vader_x_vol5` | `v[t]*s5[t]` | 5 |
| e_interaction | `fe_neg_x_vol5` | `vn[t]*s5[t]` | 5 |
| e_interaction | `fe_news_x_vol5` | `ln(1+n[t])*s5[t]` | 5 |
| f_regime | `fe_vol5` | `std(r[t-4..t])` | 5 |
| f_regime | `fe_vol_ratio_5_20` | `clip(ln((s5+1e-6)/(s20+1e-6)), -3, 3)` | 20 |
| f_regime | `fe_vol_high` | `1[s5[t] > s20[t]]` | 20 |
| f_regime | `fe_is_monday` | `1[dayofweek=0]` | 0 |
| f_regime | `fe_is_friday` | `1[dayofweek=4]` | 0 |
| g_price | `fe_ret_t` | `r[t]` | 1 |
| g_price | `fe_ret_z5` | `(r[t]-mean(r[t-4..t-1]))/(std+1e-6)` | 5 |
| g_price | `fe_rsi5` | `mean(1[r[t-4..t]>0])` | 5 |

Rolling std memakai `min_periods`; di warm-up awal diganti expanding std (tetap causal),
sisanya diisi 0. Total 27 fitur, semua lolos uji leakage.

## 2. Hari tanpa berita

`sentiment_daily` mengisi 0 untuk hari tanpa berita: 126 dari 1.202 baris sentimen,
dan 102 dari 836 baris train (12,2%; val 15/178, test 4/180).
0 itu placeholder "netral", bukan skor asli. Perlakuan:

- level, EWM, interaksi: 0 dipakai apa adanya (netral) — EWM ikut meluruh ke 0.
- delta (`d1`) dan surprise: dipaksa 0 saat `n[t]=0`, supaya hari kosong tidak
  dianggap "sentimen jatuh ke nol".
- dispersi/range: memakai deret observed-only (`v.where(n>0)`) sehingga hari kosong
  tidak menekan std ke 0; minimal 2 observasi, kalau tidak ada diisi 0.
- `fe_no_news` memberi model jalan membedakan "netral karena berita netral" dari
  "tidak ada berita".

## 3. Uji anti-leakage

Dua uji, 42 tanggal sampel (train + val, tidak termasuk test):

1. **Truncation**: data dipotong di `t`, fitur dihitung ulang, nilai baris `t`
   dibandingkan dengan hasil pipeline penuh.
2. **Future shock**: semua kolom numerik setelah `t` dikalikan 1,7 lalu ditambah 0,123;
   fitur baris `t` harus tidak berubah.

Hasil: 2.268 pemeriksaan, **0 gagal**, selisih maksimum 0,0
(`outputs_fe/leakage_summary.csv`, `outputs_fe/leakage_checks.csv`). Tidak ada fitur
yang dibuang.

Catatan desain: baseline z-score memakai `t-20..t-1` (bukan termasuk `t`), EWM causal,
ambang `fe_vol_high` dari perbandingan `s5` vs `s20` (bukan kuartil yang di-fit di train),
dan tidak ada parameter yang di-fit saat membangun fitur. Scaler hanya dipakai di tahap
screening dan di-fit pada train.

## 4. Penyaringan

Semua angka di bawah dari **val** (train untuk fit scaler). Test tidak dipakai.

**Coverage / NaN.** 27 fitur, 0 NaN setelah konstruksi (`outputs_fe/feature_coverage.csv`,
81 baris = 27 fitur × 3 split). Tidak ada fitur konstan di train.

**Korelasi terhadap target (train).** Semua lemah; terkuat `fe_ret_z5` Pearson 0,107,
lalu `fe_is_friday` −0,097, `fe_ret_t` 0,106, `fe_is_monday` 0,056,
`fe_title_text_gap` 0,057 (`outputs_fe/correlation_train.csv`). Tidak ada fitur yang
layak dipakai sendirian.

**Per kelompok** — Model A + satu kelompok, val RMSE, Δ% terhadap A (dan naive 0):

| Kelompok | n | Δ Ridge vs A | Δ LGBM vs A | Δ Ridge vs naive0 | Δ LGBM vs naive0 | Nilai |
|---|---|---|---|---|---|---|
| a_news | 4 | −0,03% | −1,51% | +0,29% | +2,66% | marginal |
| b_momentum | 6 | −1,38% | −1,63% | −1,06% | +2,54% | **terbaik** |
| c_decay | 3 | +0,46% | −2,26% | +0,78% | +1,88% | campuran |
| d_surprise | 3 | +0,51% | +1,63% | +0,83% | +5,94% | **tidak menambah** |
| e_interaction | 3 | +0,75% | +3,07% | +1,07% | +7,43% | **tidak menambah** |
| f_regime | 5 | +0,77% | −0,47% | +1,09% | +3,74% | marginal |
| g_price | 3 | −0,39% | +1,69% | −0,08% | +6,00% | campuran |

Sumber: `outputs_fe/group_screen.csv`. Hanya `b_momentum` yang memperbaiki RMSE di
kedua model. `A + semua 27 fitur` malah merugikan LightGBM (+0,64%) walau Ridge
turun 0,88% — menumpuk semua fitur bukan strategi yang benar.

**Katalog besar.** 1.701 kombinasi × 2 model = 3.402 fit di train/val
(`outputs_fe/combo_screen_wide.csv`): 127 subset grup, 27 single, 351 pasangan fitur,
1.196 subset acak, 7 drop-one-group. Hasil (`outputs_fe/wide_summary.json`):

- 601 kombinasi mengalahkan A di **kedua** model.
- 22 kombinasi mengalahkan naive 0 di **kedua** model (1,3% dari katalog).
- 19 dari 22 itu subset acak → kemungkinan besar overfit val; hanya 2 subset grup dan
  1 pasangan fitur yang lolos. Ini alasan seleksi akhir dibatasi ke subset grup.

**Kolom final** (`outputs_fe/final_columns.csv`, `final_selection.json`): `w0066` =
a_news + b_momentum + c_decay + g_price, 23 kolom (7 A + 16 FE).

| Model | val RMSE | Δ vs A | Δ vs naive0 | DirAcc |
|---|---|---|---|---|
| Ridge | 0,0033662 | −0,78% | −0,46% | 0,489 |
| LightGBM | 0,0033699 | −4,40% | −0,35% | 0,563 |

Kejujuran hasil: perbaikan terhadap naive 0 tipis (0,35–0,46%), dan CV time-series
di train lebih buruk daripada val (Ridge 0,00458 vs 0,00337) karena rezim 2022–2023
lebih volatil. Tidak ada klaim bahwa ini menang di test.

## 5. Next (tahap MLP)

Kolom final, urutan ini (`outputs_fe/final_columns.csv`), total 23:

```
kurs_lag1, kurs_lag2, kurs_lag3, kurs_lag4, kurs_lag5,
kurs_roll_mean5, kurs_roll_std5,
fe_news_count, fe_no_news, fe_news_ma5, fe_news_z20,
fe_vader_d1, fe_vader_d3, fe_lmnet_d1, fe_vader_dailystd5,
fe_vader_range5d, fe_title_text_gap,
fe_vader_ewm3, fe_vader_ewm10, fe_lmnet_ewm5,
fe_ret_t, fe_ret_z5, fe_rsi5
```

Catatan untuk tahap MLP: scaler di-fit di train saja (seperti screening); fitur level
kurs (`kurs_lag*`, `kurs_roll_mean5`) tidak stasioner, jadi pertimbangkan return-based
saja sebagai ablasi. Arsitektur belum dirancang di sini.

## 6. Isi RUN_DIR

```
fe_pipeline.py                     # features / screen / select / wide / select_wide
fe_pipeline.ipynb                  # versi notebook, kode sama dipecah per fungsi
features/features_{train,val,test}.parquet
outputs_fe/leakage_summary.csv     # 0 gagal dari 2.268 cek
outputs_fe/feature_coverage.csv
outputs_fe/correlation_train.csv
outputs_fe/group_screen.csv        # tabel per kelompok di atas
outputs_fe/combo_screen_wide.csv   # 3.402 baris
outputs_fe/combo_rank_wide.csv     # 1.701 kombinasi teragregasi
outputs_fe/wide_summary.json
outputs_fe/wide_cv.csv
outputs_fe/final_columns.csv
outputs_fe/final_selection.json
```

