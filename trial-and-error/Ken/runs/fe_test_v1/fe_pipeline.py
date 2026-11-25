"""FE test v1 - feature engineering untuk kurs USD/IDR + berita.

Dua tahap:
  python fe_pipeline.py --stage features   # bangun fitur + uji anti-leakage
  python fe_pipeline.py --stage screen     # banyak kombinasi fitur di train/val
  python fe_pipeline.py --stage all

Aturan:
- Fitur hari t hanya memakai data <= t.
- Tidak ada parameter yang di-fit saat membangun fitur (scaler hanya di screening).
- Test set tidak dipakai untuk memilih fitur.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SEED = 42
TOL_TRUNC = 1e-9
TOL_SHOCK = 1e-8


def find_root(start: Path) -> Path:
    for cand in [start.resolve(), *start.resolve().parents]:
        if (cand / "data" / "export" / "prices_daily.parquet").exists():
            return cand
    raise FileNotFoundError("Folder proyek dengan data/export/prices_daily.parquet tidak ditemukan")


RUN_DIR = Path(__file__).resolve().parent
ROOT = find_root(RUN_DIR)
EXPORT_DIR = ROOT / "data" / "export"
FEATURE_DIR = RUN_DIR / "features"
OUT_DIR = RUN_DIR / "outputs_fe"
sys.path.insert(0, str((ROOT / "src").resolve()))
from eval_utils import evaluate  # noqa: E402

BASELINE_A = [
    "kurs_lag1", "kurs_lag2", "kurs_lag3", "kurs_lag4", "kurs_lag5",
    "kurs_roll_mean5", "kurs_roll_std5",
]

# group, feature, formula, lookback, leakage, hypothesis
FEATURE_SPECS = [
    # a. aliran berita
    dict(group="a_news", feature="fe_news_count", formula="news_count[t]", lookback=0,
         leakage="rendah; artikel sudah dipetakan ke hari trading t (cutoff 15:00 WIB)",
         hypothesis="volume berita = atensi; ekstrem -> volatilitas naik"),
    dict(group="a_news", feature="fe_no_news", formula="1[news_count[t] == 0]", lookback=0,
         leakage="rendah; flag tanggal t",
         hypothesis="hari tanpa berita beda rezim; 0 sentimen adalah placeholder"),
    dict(group="a_news", feature="fe_news_ma5", formula="mean(news_count[t-4..t])", lookback=5,
         leakage="rendah; window berakhir di t",
         hypothesis="arus berita halus, mengurangi noise harian"),
    dict(group="a_news", feature="fe_news_z20",
         formula="(n[t]-mean(n[t-20..t-1]))/(std(n[t-20..t-1])+1)", lookback=21,
         leakage="rendah; baseline tidak memasukkan t",
         hypothesis="lonjakan berita = kejutan informasi"),
    # b. momentum & dispersi sentimen
    dict(group="b_momentum", feature="fe_vader_d1",
         formula="0 if n[t]=0 else v[t]-v_ffill[t-1]", lookback=1,
         leakage="rendah; ffill hanya masa lalu",
         hypothesis="perubahan sentimen judul = berita baru"),
    dict(group="b_momentum", feature="fe_vader_d3",
         formula="mean(v_obs[t-2..t])-mean(v_obs[t-5..t-3])", lookback=5,
         leakage="rendah; hanya observasi masa lalu/t",
         hypothesis="momentum sentimen 3 hari vs 3 hari sebelumnya"),
    dict(group="b_momentum", feature="fe_lmnet_d1",
         formula="0 if n[t]=0 else l[t]-l_ffill[t-1]", lookback=1,
         leakage="rendah; ffill hanya masa lalu",
         hypothesis="perubahan nada isi artikel"),
    dict(group="b_momentum", feature="fe_vader_dailystd5",
         formula="std(v_obs[t-4..t]), min 2 observasi, else 0", lookback=5,
         leakage="rendah; hanya observasi masa lalu/t",
         hypothesis="dispersi sentimen harian = ketidaksepakatan"),
    dict(group="b_momentum", feature="fe_vader_range5d",
         formula="max(v_obs[t-4..t])-min(v_obs[t-4..t]), else 0", lookback=5,
         leakage="rendah; hanya observasi masa lalu/t",
         hypothesis="rentang ekstrem sentimen"),
    dict(group="b_momentum", feature="fe_title_text_gap",
         formula="lm_title_net[t]-lm_text_net[t]", lookback=0,
         leakage="rendah; keduanya hari t",
         hypothesis="judul sensasional vs isi artikel"),
    # c. decay eksponensial
    dict(group="c_decay", feature="fe_vader_ewm3",
         formula="EWM(v[t], halflife=3, adjust=False)", lookback="inf (efektif ~9 hari)",
         leakage="rendah; EWM causal",
         hypothesis="reaksi cepat ke berita terbaru"),
    dict(group="c_decay", feature="fe_vader_ewm10",
         formula="EWM(v[t], halflife=10, adjust=False)", lookback="inf (efektif ~30 hari)",
         leakage="rendah; EWM causal",
         hypothesis="sentimen lambat/persisten"),
    dict(group="c_decay", feature="fe_lmnet_ewm5",
         formula="EWM(l[t], halflife=5, adjust=False)", lookback="inf (efektif ~15 hari)",
         leakage="rendah; EWM causal",
         hypothesis="nada isi yang bertahan"),
    # d. surprise
    dict(group="d_surprise", feature="fe_vader_surprise20",
         formula="0 if n[t]=0 else (v[t]-mean(v[t-20..t-1]))/(std(...)+0.01)", lookback=21,
         leakage="rendah; baseline tidak memasukkan t",
         hypothesis="kejutan sentimen vs kebiasaan 20 hari"),
    dict(group="d_surprise", feature="fe_lmnet_surprise20",
         formula="0 if n[t]=0 else (l[t]-mean(l[t-20..t-1]))/(std(...)+0.01)", lookback=21,
         leakage="rendah; baseline tidak memasukkan t",
         hypothesis="kejutan nada isi"),
    dict(group="d_surprise", feature="fe_lmunc_surprise20",
         formula="0 if n[t]=0 else (u[t]-mean(u[t-20..t-1]))/(std(...)+0.01)", lookback=21,
         leakage="rendah; baseline tidak memasukkan t",
         hypothesis="lonjakan ketidakpastian = risiko"),
    # e. interaksi teks x volatilitas
    dict(group="e_interaction", feature="fe_vader_x_vol5",
         formula="v[t]*sigma5[t]", lookback=5,
         leakage="rendah; sigma5 dari return t-4..t",
         hypothesis="sentimen lebih relevan saat kurs volatil"),
    dict(group="e_interaction", feature="fe_neg_x_vol5",
         formula="vader_title_neg_ratio[t]*sigma5[t]", lookback=5,
         leakage="rendah; sigma5 dari return t-4..t",
         hypothesis="berita negatif x volatilitas = tekanan"),
    dict(group="e_interaction", feature="fe_news_x_vol5",
         formula="ln(1+news_count[t])*sigma5[t]", lookback=5,
         leakage="rendah; sigma5 dari return t-4..t",
         hypothesis="arus berita x volatilitas"),
    # f. rezim / kalender
    dict(group="f_regime", feature="fe_vol5",
         formula="std(ret[t-4..t])", lookback=5,
         leakage="rendah; return t sudah diketahui",
         hypothesis="volatilitas jangka pendek"),
    dict(group="f_regime", feature="fe_vol_ratio_5_20",
         formula="clip(ln((sigma5[t]+1e-6)/(sigma20[t]+1e-6)), -3, 3)", lookback=20,
         leakage="rendah; kedua window berakhir di t",
         hypothesis="vol pendek relatif vol panjang (log ratio)"),
    dict(group="f_regime", feature="fe_vol_high",
         formula="1[sigma5[t] > sigma20[t]]", lookback=20,
         leakage="rendah; ambang tidak di-fit",
         hypothesis="rezim volatil tinggi"),
    dict(group="f_regime", feature="fe_is_monday",
         formula="1[dayofweek[t] == 0]", lookback=0,
         leakage="rendah; kalender",
         hypothesis="efek Senin / berita akhir pekan"),
    dict(group="f_regime", feature="fe_is_friday",
         formula="1[dayofweek[t] == 4]", lookback=0,
         leakage="rendah; kalender",
         hypothesis="efek Jumat / posisi akhir pekan"),
    # g. fitur kurs non-linier yang belum ada di A
    dict(group="g_price", feature="fe_ret_t",
         formula="ret[t] = ln(kurs_mid[t]/kurs_mid[t-1])", lookback=1,
         leakage="rendah; A tidak punya level t",
         hypothesis="return hari ini, bukan duplikat lag A"),
    dict(group="g_price", feature="fe_ret_z5",
         formula="(ret[t]-mean(ret[t-4..t-1]))/(std(ret[t-4..t-1])+1e-6)", lookback=5,
         leakage="rendah; baseline tidak memasukkan t",
         hypothesis="kejutan return hari ini"),
    dict(group="g_price", feature="fe_rsi5",
         formula="mean(1[ret[t-4..t] > 0])", lookback=5,
         leakage="rendah; window berakhir di t",
         hypothesis="persistensi arah return"),
]

FEATURES = [s["feature"] for s in FEATURE_SPECS]
FEATURE_GROUPS: dict[str, list[str]] = {}
for _spec in FEATURE_SPECS:
    FEATURE_GROUPS.setdefault(_spec["group"], []).append(_spec["feature"])


def load_raw() -> pd.DataFrame:
    prices = pd.read_parquet(EXPORT_DIR / "prices_daily.parquet")
    sent = pd.read_parquet(EXPORT_DIR / "sentiment_daily.parquet")
    for part in (prices, sent):
        part["date"] = pd.to_datetime(part["date"]).dt.normalize()
    df = prices.merge(sent, on="date", how="left", validate="one_to_one")
    df = df.sort_values("date").reset_index(drop=True)
    if df[["news_count", "vader_title_mean", "lm_text_net"]].isna().any().any():
        raise ValueError("Ada tanggal kurs tanpa baris sentimen")
    return df


def _rolling_std_causal(x: pd.Series, window: int, min_exp: int) -> pd.Series:
    """Rolling std dengan fallback expanding di warm-up; keduanya causal."""
    roll = x.rolling(window, min_periods=window).std()
    exp = x.expanding(min_periods=min_exp).std()
    return roll.fillna(exp).fillna(0.0)


def _surprise(x: pd.Series, mask_news: pd.Series, window: int = 20,
              floor: float = 0.01, min_periods: int = 5) -> pd.Series:
    base = x.shift(1).rolling(window, min_periods=min_periods)
    z = (x - base.mean()) / (base.std() + floor)
    return z.where(mask_news, 0.0).fillna(0.0)


def compute_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hitung fitur untuk seluruh baris df. Tidak memakai target dan tidak melihat masa depan."""
    out = df[["date", "split", "target", "ret", "kurs_mid", *BASELINE_A]].copy()

    n = df["news_count"].astype(float)
    v = df["vader_title_mean"].astype(float)
    l = df["lm_text_net"].astype(float)
    u = df["lm_text_uncertainty"].astype(float)
    vn = df["vader_title_neg_ratio"].astype(float)
    lt = df["lm_title_net"].astype(float)
    r = df["ret"].astype(float)
    mask_news = n > 0

    # a. aliran berita
    out["fe_news_count"] = n
    out["fe_no_news"] = (n == 0).astype(float)
    out["fe_news_ma5"] = n.rolling(5, min_periods=1).mean()
    base_n = n.shift(1).rolling(20, min_periods=5)
    out["fe_news_z20"] = ((n - base_n.mean()) / (base_n.std() + 1.0)).fillna(0.0)

    # b. momentum & dispersi sentimen (observed-only)
    v_obs = v.where(mask_news)
    l_obs = l.where(mask_news)
    v_ffill = v_obs.ffill()
    l_ffill = l_obs.ffill()
    out["fe_vader_d1"] = (v_obs - v_ffill.shift(1)).where(mask_news, 0.0).fillna(0.0)
    out["fe_vader_d3"] = (
        v_obs.rolling(3, min_periods=1).mean()
        - v_obs.shift(3).rolling(3, min_periods=1).mean()
    ).fillna(0.0)
    out["fe_lmnet_d1"] = (l_obs - l_ffill.shift(1)).where(mask_news, 0.0).fillna(0.0)
    out["fe_vader_dailystd5"] = v_obs.rolling(5, min_periods=2).std().fillna(0.0)
    out["fe_vader_range5d"] = (
        v_obs.rolling(5, min_periods=1).max()
        - v_obs.rolling(5, min_periods=1).min()
    ).fillna(0.0)
    out["fe_title_text_gap"] = lt - l

    # c. decay eksponensial (0 = netral, jadi hari tanpa berita ikut meluruh)
    out["fe_vader_ewm3"] = v.ewm(halflife=3, adjust=False).mean()
    out["fe_vader_ewm10"] = v.ewm(halflife=10, adjust=False).mean()
    out["fe_lmnet_ewm5"] = l.ewm(halflife=5, adjust=False).mean()

    # d. surprise vs baseline rolling (baseline tidak termasuk t)
    out["fe_vader_surprise20"] = _surprise(v, mask_news, 20, 0.01)
    out["fe_lmnet_surprise20"] = _surprise(l, mask_news, 20, 0.01)
    out["fe_lmunc_surprise20"] = _surprise(u, mask_news, 20, 0.01)

    # e/f. volatilitas return + interaksi
    vol5 = _rolling_std_causal(r, 5, min_exp=2)
    vol20 = _rolling_std_causal(r, 20, min_exp=5)
    out["fe_vol5"] = vol5
    out["fe_vol_ratio_5_20"] = np.clip(np.log((vol5 + 1e-6) / (vol20 + 1e-6)), -3.0, 3.0)
    out["fe_vol_high"] = (vol5 > vol20).astype(float)
    out["fe_vader_x_vol5"] = v * vol5
    out["fe_neg_x_vol5"] = vn * vol5
    out["fe_news_x_vol5"] = np.log1p(n) * vol5

    # f. kalender
    dow = df["date"].dt.dayofweek
    out["fe_is_monday"] = (dow == 0).astype(float)
    out["fe_is_friday"] = (dow == 4).astype(float)

    # g. kurs non-linier
    out["fe_ret_t"] = r
    base_r = r.shift(1).rolling(5, min_periods=3)
    out["fe_ret_z5"] = ((r - base_r.mean()) / (base_r.std() + 1e-6)).fillna(0.0)
    out["fe_rsi5"] = (r > 0).astype(float).rolling(5, min_periods=1).mean()

    missing = [f for f in FEATURES if f not in out.columns]
    if missing:
        raise RuntimeError(f"Fitur belum dihitung: {missing}")

    nan_rows = []
    for f in FEATURES:
        col = out[f]
        nan_rows.append({
            "feature": f,
            "nan_total": int(col.isna().sum()),
            "nan_train": int(col[out["split"] == "train"].isna().sum()),
            "nan_val": int(col[out["split"] == "val"].isna().sum()),
            "nan_test": int(col[out["split"] == "test"].isna().sum()),
        })
    nan_df = pd.DataFrame(nan_rows)

    for f in FEATURES:
        out[f] = out[f].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, nan_df


def sample_indices(df: pd.DataFrame, step_train: int = 29, step_val: int = 17) -> list[int]:
    """Sampel hanya train/val; test tidak dipakai untuk cek/seleksi."""
    idx: list[int] = []
    for split, step in (("train", step_train), ("val", step_val)):
        pos = np.flatnonzero(df["split"].values == split)
        if len(pos):
            idx.extend([int(pos[0]), int(pos[-1])])
            idx.extend(int(x) for x in pos[::step])
    return sorted(set(idx))


def run_truncation(raw: pd.DataFrame, full: pd.DataFrame, idxs: list[int]) -> pd.DataFrame:
    rows = []
    for idx in idxs:
        t_date = raw.loc[idx, "date"]
        trunc = raw.loc[raw["date"] <= t_date].copy()
        feat_trunc, _ = compute_features(trunc)
        a = full.loc[full["date"] == t_date, FEATURES].iloc[0]
        b = feat_trunc.loc[feat_trunc["date"] == t_date, FEATURES].iloc[0]
        for f in FEATURES:
            diff = abs(float(a[f]) - float(b[f]))
            rows.append({"test": "truncation", "date": t_date, "feature": f,
                         "abs_diff": diff, "pass": bool(diff <= TOL_TRUNC)})
    return pd.DataFrame(rows)


def run_future_shock(raw: pd.DataFrame, full: pd.DataFrame, idxs: list[int]) -> pd.DataFrame:
    num_cols = [c for c in raw.columns
                if c not in ("date", "split") and pd.api.types.is_numeric_dtype(raw[c])]
    rows = []
    for idx in idxs:
        t_date = raw.loc[idx, "date"]
        shocked = raw.copy()
        future = shocked["date"] > t_date
        for c in num_cols:
            shocked.loc[future, c] = shocked.loc[future, c].astype(float) * 1.7 + 0.123
        feat_shock, _ = compute_features(shocked)
        a = full.loc[full["date"] == t_date, FEATURES].iloc[0]
        b = feat_shock.loc[feat_shock["date"] == t_date, FEATURES].iloc[0]
        for f in FEATURES:
            diff = abs(float(a[f]) - float(b[f]))
            rows.append({"test": "future_shock", "date": t_date, "feature": f,
                         "abs_diff": diff, "pass": bool(diff <= TOL_SHOCK)})
    return pd.DataFrame(rows)


def summarize_checks(checks: pd.DataFrame) -> pd.DataFrame:
    out = (checks.groupby(["test", "feature"])
           .agg(n_checks=("date", "size"),
                max_abs_diff=("abs_diff", "max"),
                n_fail=("pass", lambda s: int((~s).sum())))
           .reset_index())
    out["pass"] = out["n_fail"] == 0
    return out


def stage_features() -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    full, nan_df = compute_features(raw)
    idxs = sample_indices(raw)
    checks = pd.concat([run_truncation(raw, full, idxs),
                        run_future_shock(raw, full, idxs)], ignore_index=True)
    summary = summarize_checks(checks)
    checks.to_csv(OUT_DIR / "leakage_checks.csv", index=False)
    summary.to_csv(OUT_DIR / "leakage_summary.csv", index=False)

    failed = sorted(summary.loc[~summary["pass"], "feature"].unique().tolist())
    kept = [f for f in FEATURES if f not in failed]

    manifest = pd.DataFrame(FEATURE_SPECS)
    manifest["status"] = np.where(manifest["feature"].isin(failed), "dropped_leakage", "kept")
    manifest.to_csv(OUT_DIR / "feature_manifest.csv", index=False)

    if failed:
        full = full.drop(columns=failed)

    coverage = []
    for split in ("train", "val", "test"):
        part = full[full["split"] == split]
        for f in kept:
            raw_nan = int(nan_df.loc[nan_df["feature"] == f, f"nan_{split}"].iloc[0])
            coverage.append({
                "split": split, "feature": f, "n_rows": len(part),
                "n_nan_raw": raw_nan, "n_nan_after_fill": int(part[f].isna().sum()),
                "mean": float(part[f].mean()), "std": float(part[f].std()),
                "min": float(part[f].min()), "max": float(part[f].max()),
            })
    pd.DataFrame(coverage).to_csv(OUT_DIR / "feature_coverage.csv", index=False)

    for split in ("train", "val", "test"):
        part = full[full["split"] == split].reset_index(drop=True)
        part.to_parquet(FEATURE_DIR / f"features_{split}.parquet", index=False)

    payload = {
        "n_rows": int(len(full)),
        "split": {k: int((full["split"] == k).sum()) for k in ("train", "val", "test")},
        "n_features_requested": len(FEATURES),
        "n_features_kept": len(kept),
        "failed_features": failed,
        "n_check_dates": len(idxs),
        "check_dates": [str(raw.loc[i, "date"].date()) for i in idxs],
        "n_checks": int(len(checks)),
        "n_check_failures": int((~checks["pass"]).sum()),
        "tolerances": {"truncation": TOL_TRUNC, "future_shock": TOL_SHOCK},
        "feature_groups": {g: [f for f in fs if f in kept] for g, fs in FEATURE_GROUPS.items()},
    }
    (OUT_DIR / "feature_stage_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: payload[k] for k in
                      ("n_rows", "split", "n_features_requested", "n_features_kept",
                       "failed_features", "n_check_dates", "n_checks", "n_check_failures")},
                     indent=2, ensure_ascii=False))


def _load_feature_splits() -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, list[str]]]:
    train = pd.read_parquet(FEATURE_DIR / "features_train.parquet")
    val = pd.read_parquet(FEATURE_DIR / "features_val.parquet")
    manifest = pd.read_csv(OUT_DIR / "feature_manifest.csv")
    kept = manifest.loc[manifest["status"] == "kept", "feature"].tolist()
    groups: dict[str, list[str]] = {}
    for _, row in manifest[manifest["status"] == "kept"].iterrows():
        groups.setdefault(row["group"], []).append(row["feature"])
    return train, val, kept, groups


def _model_factories():
    from sklearn.linear_model import Ridge
    from lightgbm import LGBMRegressor

    return {
        "Ridge": lambda: Ridge(alpha=1.0),
        "LightGBM": lambda: LGBMRegressor(
            n_estimators=300, learning_rate=0.03, num_leaves=7,
            min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=SEED, n_jobs=2,
            verbosity=-1, force_col_wise=True,
        ),
    }


def _fit_eval(train: pd.DataFrame, val: pd.DataFrame, cols: list[str],
              factory, seed: int = SEED) -> dict:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train[cols])
    model = factory()
    model.fit(scaler.transform(train[cols]), train["target"].values)
    pred = model.predict(scaler.transform(val[cols]))
    return evaluate(val["target"].values, pred)


def _cv_rmse(train: pd.DataFrame, cols: list[str], factory, n_splits: int = 5) -> tuple[float, float]:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr_idx, va_idx in tscv.split(train):
        tr = train.iloc[tr_idx]
        va = train.iloc[va_idx]
        scaler = StandardScaler().fit(tr[cols])
        model = factory().fit(scaler.transform(tr[cols]), tr["target"].values)
        pred = model.predict(scaler.transform(va[cols]))
        scores.append(evaluate(va["target"].values, pred)["RMSE"])
    return float(np.mean(scores)), float(np.std(scores))


def _combo_catalog(kept: list[str], groups: dict[str, list[str]]) -> list[dict]:
    group_names = list(groups)
    catalog: list[dict] = []

    for r in range(1, len(group_names) + 1):
        for gs in combinations(group_names, r):
            feats = tuple(f for g in gs for f in groups[g])
            catalog.append({"combo_type": "group_subset", "groups": gs, "features": feats})

    for f in kept:
        catalog.append({"combo_type": "single_feature", "groups": (), "features": (f,)})

    rng = np.random.default_rng(SEED)
    for _ in range(120):
        k = int(rng.integers(2, min(16, len(kept)) + 1))
        feats = tuple(sorted(rng.choice(kept, size=k, replace=False).tolist()))
        catalog.append({"combo_type": "random_features", "groups": (), "features": feats})

    for g in group_names:
        gs = tuple(x for x in group_names if x != g)
        feats = tuple(f for x in gs for f in groups[x])
        catalog.append({"combo_type": "drop_one_group", "groups": gs, "features": feats})

    seen = set()
    unique = []
    for item in catalog:
        key = item["features"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    for i, item in enumerate(unique):
        item["combo_id"] = f"c{i:03d}"
    return unique


def stage_screen() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, val, kept, groups = _load_feature_splits()
    models = _model_factories()

    y_val = val["target"].values
    naive_zero = evaluate(y_val, np.zeros(len(y_val)))

    corr_rows = []
    for f in kept:
        corr_rows.append({
            "feature": f,
            "pearson": float(train[f].corr(train["target"])),
            "spearman": float(train[f].corr(train["target"], method="spearman")),
        })
    corr = pd.DataFrame(corr_rows).sort_values("spearman", key=lambda s: s.abs(), ascending=False)
    corr.to_csv(OUT_DIR / "correlation_train.csv", index=False)

    # baseline A
    base_metrics = {}
    base_rows = []
    for name, factory in models.items():
        m = _fit_eval(train, val, BASELINE_A, factory)
        base_metrics[name] = m
        base_rows.append({"combo_id": "baseline_A", "combo_type": "baseline",
                          "groups": "", "n_new_features": 0,
                          "n_total_features": len(BASELINE_A), "model": name, **m})
    pd.DataFrame(base_rows).to_csv(OUT_DIR / "baseline_A_val.csv", index=False)

    catalog = _combo_catalog(kept, groups)
    rows = []
    for item in catalog:
        cols = BASELINE_A + list(item["features"])
        for name, factory in models.items():
            m = _fit_eval(train, val, cols, factory)
            rows.append({
                "combo_id": item["combo_id"],
                "combo_type": item["combo_type"],
                "groups": "+".join(item["groups"]),
                "n_new_features": len(item["features"]),
                "n_total_features": len(cols),
                "model": name,
                **m,
                "delta_rmse_pct_vs_A": 100 * (m["RMSE"] - base_metrics[name]["RMSE"]) / base_metrics[name]["RMSE"],
                "delta_rmse_pct_vs_naive0": 100 * (m["RMSE"] - naive_zero["RMSE"]) / naive_zero["RMSE"],
            })
    screen = pd.DataFrame(rows)
    screen.to_csv(OUT_DIR / "combo_screen.csv", index=False)

    top = (screen.sort_values(["model", "RMSE"])
           .groupby("model", group_keys=False).head(20).reset_index(drop=True))
    top.to_csv(OUT_DIR / "top_combos_val.csv", index=False)

    # per kelompok: A + group, A + all, A + all minus group
    group_rows = []
    for name, factory in models.items():
        m = _fit_eval(train, val, BASELINE_A, factory)
        group_rows.append({"variant": "A", "group": "", "n_new_features": 0,
                           "model": name, **m})
        for g, feats in groups.items():
            m = _fit_eval(train, val, BASELINE_A + feats, factory)
            group_rows.append({"variant": "A+group", "group": g,
                               "n_new_features": len(feats), "model": name, **m})
        all_feats = [f for fs in groups.values() for f in fs]
        m = _fit_eval(train, val, BASELINE_A + all_feats, factory)
        group_rows.append({"variant": "A+all", "group": "all",
                           "n_new_features": len(all_feats), "model": name, **m})
        for g, feats in groups.items():
            drop = [f for f in all_feats if f not in feats]
            m = _fit_eval(train, val, BASELINE_A + drop, factory)
            group_rows.append({"variant": "A+all-minus-group", "group": g,
                               "n_new_features": len(drop), "model": name, **m})
    group_screen = pd.DataFrame(group_rows)
    for name in models:
        base = group_screen[(group_screen["model"] == name) & (group_screen["variant"] == "A")]["RMSE"].iloc[0]
        group_screen.loc[group_screen["model"] == name, "delta_rmse_pct_vs_A"] = (
            100 * (group_screen.loc[group_screen["model"] == name, "RMSE"] - base) / base
        )
    group_screen.to_csv(OUT_DIR / "group_screen.csv", index=False)

    # cek stabilitas CV untuk 10 kombinasi teratas per model
    cv_rows = []
    for name, factory in models.items():
        for _, row in top[top["model"] == name].head(10).iterrows():
            item = next(c for c in catalog if c["combo_id"] == row["combo_id"])
            cols = BASELINE_A + list(item["features"])
            mean_cv, std_cv = _cv_rmse(train, cols, factory)
            cv_rows.append({"combo_id": row["combo_id"], "model": name,
                            "val_RMSE": row["RMSE"], "cv_RMSE_mean": mean_cv,
                            "cv_RMSE_std": std_cv})
    pd.DataFrame(cv_rows).to_csv(OUT_DIR / "top_combo_cv.csv", index=False)

    best = {}
    for name in models:
        sub = screen[screen["model"] == name].sort_values("RMSE").iloc[0]
        best[name] = {"combo_id": sub["combo_id"], "combo_type": sub["combo_type"],
                      "groups": sub["groups"], "n_total_features": int(sub["n_total_features"]),
                      "RMSE": float(sub["RMSE"]), "MAE": float(sub["MAE"]),
                      "DirAcc": float(sub["DirAcc"]),
                      "delta_rmse_pct_vs_A": float(sub["delta_rmse_pct_vs_A"])}

    payload = {
        "n_combos": len(catalog),
        "n_new_features_kept": len(kept),
        "naive0_val": naive_zero,
        "baseline_A_val": base_metrics,
        "best_combo_per_model": best,
        "caveat": "Pemeringkatan memakai val; test tidak dipakai. Top val bisa overfit val.",
    }
    (OUT_DIR / "screening_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def stage_select() -> None:
    """Pilih subset grup terbaik berdasarkan rata-rata val RMSE Ridge+LightGBM."""
    manifest = pd.read_csv(OUT_DIR / "feature_manifest.csv")
    screen = pd.read_csv(OUT_DIR / "combo_screen.csv")
    group_only = screen[screen["combo_type"] == "group_subset"].copy()
    mean_rmse = group_only.groupby("combo_id")["RMSE"].mean()
    best_id = str(mean_rmse.idxmin())
    best = group_only[group_only["combo_id"] == best_id].iloc[0]
    groups_str = [g for g in str(best["groups"]).split("+") if g]
    group_map = (manifest[manifest["status"] == "kept"]
                 .groupby("group")["feature"].apply(list).to_dict())
    feats = [f for g in groups_str for f in group_map[g]]
    cols = BASELINE_A + feats

    rows = []
    for i, c in enumerate(cols, 1):
        if c in BASELINE_A:
            rows.append({"order": i, "column": c, "source": "A", "group": "baseline_A"})
        else:
            grp = next(g for g in groups_str if c in group_map[g])
            rows.append({"order": i, "column": c, "source": "FE", "group": grp})
    pd.DataFrame(rows).to_csv(OUT_DIR / "final_columns.csv", index=False)

    metrics = (group_only[group_only["combo_id"] == best_id]
               [["model", "RMSE", "MAE", "DirAcc",
                 "delta_rmse_pct_vs_A", "delta_rmse_pct_vs_naive0"]]
               .to_dict(orient="records"))
    payload = {
        "combo_id": best_id,
        "groups": groups_str,
        "n_features": len(cols),
        "criterion": "rata-rata val RMSE Ridge+LightGBM terbaik di antara semua subset grup",
        "val_metrics": metrics,
        "caveat": ("Dipilih di val; test tidak dipakai. Kombinasi random-feature "
                   "dengan val lebih rendah tidak dipilih karena risiko overfit val."),
    }
    (OUT_DIR / "final_selection.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _wide_catalog(kept: list[str], groups: dict[str, list[str]]) -> list[dict]:
    """Katalog besar: semua subset grup, semua single, semua pasangan, subset acak."""
    group_names = list(groups)
    catalog: list[dict] = []
    for r in range(1, len(group_names) + 1):
        for gs in combinations(group_names, r):
            feats = tuple(f for g in gs for f in groups[g])
            catalog.append({"combo_type": "group_subset", "groups": gs, "features": feats})
    for f in kept:
        catalog.append({"combo_type": "single_feature", "groups": (), "features": (f,)})
    for a, b in combinations(kept, 2):
        catalog.append({"combo_type": "feature_pair", "groups": (), "features": (a, b)})
    rng = np.random.default_rng(SEED)
    for _ in range(1200):
        k = int(rng.integers(3, min(20, len(kept)) + 1))
        feats = tuple(sorted(rng.choice(kept, size=k, replace=False).tolist()))
        catalog.append({"combo_type": "random_features", "groups": (), "features": feats})
    for g in group_names:
        gs = tuple(x for x in group_names if x != g)
        feats = tuple(f for x in gs for f in groups[x])
        catalog.append({"combo_type": "drop_one_group", "groups": gs, "features": feats})

    seen, unique = set(), []
    for item in catalog:
        if item["features"] in seen:
            continue
        seen.add(item["features"])
        unique.append(item)
    for i, item in enumerate(unique):
        item["combo_id"] = f"w{i:04d}"
    return unique


WIDE_COLS = [
    "combo_id", "combo_type", "groups", "features", "n_new_features",
    "n_total_features", "model", "MAE", "MSE", "RMSE", "DirAcc", "PredStd",
    "delta_rmse_pct_vs_A", "delta_rmse_pct_vs_naive0",
]


def _repair_wide_raw(raw_path: Path) -> None:
    """Perbaiki file mentah yang header-nya hanya 'combo_id' (bug penulisan awal)."""
    if not raw_path.exists():
        return
    with open(raw_path, encoding="utf-8") as fh:
        first = fh.readline().strip()
    if first != "combo_id":
        return
    df = pd.read_csv(raw_path, header=None, skiprows=1, names=WIDE_COLS,
                     on_bad_lines="skip")
    bad = df["model"].isna().sum()
    df = df.dropna(subset=["model"]).reset_index(drop=True)
    df.to_csv(raw_path, index=False)
    print(f"repair raw: {len(df)} baris dipulihkan, {bad} baris rusak dibuang")


def stage_wide() -> None:
    """Uji katalog besar (ribuan kombinasi) di train/val. Test tidak disentuh."""
    import gc

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, val, kept, groups = _load_feature_splits()
    models = _model_factories()
    y_val = val["target"].values
    naive_zero = evaluate(y_val, np.zeros(len(y_val)))

    base = {name: _fit_eval(train, val, BASELINE_A, f) for name, f in models.items()}

    catalog = _wide_catalog(kept, groups)
    raw_path = OUT_DIR / "combo_screen_wide_raw.csv"
    _repair_wide_raw(raw_path)
    done = set()
    if raw_path.exists():
        prev = pd.read_csv(raw_path, usecols=["combo_id"])
        done = set(prev["combo_id"].unique())
        print(f"resume: {len(done)} kombinasi sudah selesai")
    else:
        pd.DataFrame(columns=WIDE_COLS).to_csv(raw_path, index=False)

    rows = []
    for k, item in enumerate(catalog, 1):
        if item["combo_id"] in done:
            continue
        cols = BASELINE_A + list(item["features"])
        for name, factory in models.items():
            m = _fit_eval(train, val, cols, factory)
            rows.append({
                "combo_id": item["combo_id"], "combo_type": item["combo_type"],
                "groups": "+".join(item["groups"]),
                "features": "|".join(item["features"]),
                "n_new_features": len(item["features"]), "n_total_features": len(cols),
                "model": name, **m,
                "delta_rmse_pct_vs_A": 100 * (m["RMSE"] - base[name]["RMSE"]) / base[name]["RMSE"],
                "delta_rmse_pct_vs_naive0": 100 * (m["RMSE"] - naive_zero["RMSE"]) / naive_zero["RMSE"],
            })
        if len(rows) >= 100:
            pd.DataFrame(rows).to_csv(raw_path, mode="a", header=False, index=False)
            rows = []
            gc.collect()
        if k % 250 == 0:
            print(f"  {k}/{len(catalog)} kombinasi", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(raw_path, mode="a", header=False, index=False)

    wide = pd.read_csv(raw_path)
    wide = wide.drop_duplicates(subset=["combo_id", "model"], keep="last")
    wide["groups"] = wide["groups"].fillna("")
    wide.to_csv(OUT_DIR / "combo_screen_wide.csv", index=False)
    print(f"katalog: {wide['combo_id'].nunique()} kombinasi, {len(wide)} baris, "
          f"model={sorted(wide['model'].unique())}")

    agg = (wide.groupby(["combo_id", "combo_type", "groups", "features",
                         "n_new_features", "n_total_features"])
           .agg(mean_RMSE=("RMSE", "mean"), max_delta_vs_A=("delta_rmse_pct_vs_A", "max"),
                max_delta_vs_naive0=("delta_rmse_pct_vs_naive0", "max"),
                n_models_better_A=("delta_rmse_pct_vs_A", lambda s: int((s < 0).sum())),
                n_models_better_naive0=("delta_rmse_pct_vs_naive0", lambda s: int((s < 0).sum())))
           .reset_index())
    agg.to_csv(OUT_DIR / "combo_rank_wide.csv", index=False)

    stats = {
        "n_combos": int(agg.shape[0]),
        "n_fits": int(wide.shape[0]),
        "n_beat_A_both_models": int((agg["n_models_better_A"] == len(models)).sum()),
        "n_beat_naive0_both_models": int((agg["n_models_better_naive0"] == len(models)).sum()),
        "n_beat_both_A_and_naive0": int(((agg["n_models_better_A"] == len(models))
                                         & (agg["n_models_better_naive0"] == len(models))).sum()),
        "naive0_val_RMSE": naive_zero["RMSE"],
        "baseline_A_val_RMSE": {k: v["RMSE"] for k, v in base.items()},
    }
    (OUT_DIR / "wide_summary.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # stabilitas CV untuk kandidat teratas: 25 terbaik mean RMSE + 25 delta teraman
    cand = pd.concat([
        agg.sort_values("mean_RMSE").head(25),
        agg.sort_values("max_delta_vs_A").head(25),
        agg[(agg["n_models_better_A"] == len(models))
            & (agg["n_models_better_naive0"] == len(models))].sort_values("mean_RMSE").head(15),
    ]).drop_duplicates("combo_id")
    cv_rows = []
    for _, row in cand.iterrows():
        feats = [f for f in str(row["features"]).split("|") if f]
        cols = BASELINE_A + feats
        for name, factory in models.items():
            mean_cv, std_cv = _cv_rmse(train, cols, factory)
            cv_rows.append({"combo_id": row["combo_id"], "model": name,
                            "val_RMSE": float(wide[(wide["combo_id"] == row["combo_id"])
                                                   & (wide["model"] == name)]["RMSE"].iloc[0]),
                            "cv_RMSE_mean": mean_cv, "cv_RMSE_std": std_cv})
    pd.DataFrame(cv_rows).to_csv(OUT_DIR / "wide_cv.csv", index=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def stage_select_wide() -> None:
    """Pilih kolom final dari katalog besar, hanya kombinasi yang lolos dua syarat."""
    train, val, kept, groups = _load_feature_splits()
    wide = pd.read_csv(OUT_DIR / "combo_screen_wide.csv")
    agg = pd.read_csv(OUT_DIR / "combo_rank_wide.csv")
    cv = pd.read_csv(OUT_DIR / "wide_cv.csv")
    n_models = wide["model"].nunique()

    eligible = agg[(agg["n_models_better_A"] == n_models)
                   & (agg["n_models_better_naive0"] == n_models)].copy()
    structural = eligible[eligible["combo_type"].isin(["group_subset", "drop_one_group"])]
    if not structural.empty:
        eligible = structural
        pool = "subset grup (interpretable)"
    else:
        pool = "semua tipe (tidak ada subset grup yang lolos)"
    if eligible.empty:
        eligible = agg.sort_values("max_delta_vs_A").head(1).copy()
        criterion = "tidak ada kombinasi yang lolos dua syarat; diambil delta vs A teraman"
    else:
        best_mean = eligible["mean_RMSE"].min()
        near = eligible[eligible["mean_RMSE"] <= best_mean * 1.003]
        eligible = near.sort_values(["n_new_features", "mean_RMSE"])
        criterion = (f"lolos: kalahkan A dan naive0 di Ridge+LightGBM pada katalog {pool}; "
                     "lalu pilih fitur paling sedikit dalam 0.3% mean RMSE terbaik")

    chosen = eligible.iloc[0]
    feats = [f for f in str(chosen["features"]).split("|") if f]
    cols = BASELINE_A + feats
    group_map = (pd.read_csv(OUT_DIR / "feature_manifest.csv")
                 .query("status == 'kept'").groupby("group")["feature"].apply(list).to_dict())
    feat_group = {f: g for g, fs in group_map.items() for f in fs}

    rows = []
    for i, c in enumerate(cols, 1):
        rows.append({"order": i, "column": c,
                     "source": "A" if c in BASELINE_A else "FE",
                     "group": "baseline_A" if c in BASELINE_A else feat_group[c]})
    pd.DataFrame(rows).to_csv(OUT_DIR / "final_columns.csv", index=False)

    metrics = wide[wide["combo_id"] == chosen["combo_id"]][
        ["model", "RMSE", "MAE", "DirAcc", "PredStd",
         "delta_rmse_pct_vs_A", "delta_rmse_pct_vs_naive0"]].to_dict(orient="records")
    cv_rows = cv[cv["combo_id"] == chosen["combo_id"]].to_dict(orient="records")
    if not cv_rows:
        cv_rows = []
        for name, factory in _model_factories().items():
            mean_cv, std_cv = _cv_rmse(train, cols, factory)
            cv_rows.append({"combo_id": chosen["combo_id"], "model": name,
                            "val_RMSE": float(wide[(wide["combo_id"] == chosen["combo_id"])
                                                   & (wide["model"] == name)]["RMSE"].iloc[0]),
                            "cv_RMSE_mean": mean_cv, "cv_RMSE_std": std_cv})
        pd.concat([cv, pd.DataFrame(cv_rows)], ignore_index=True).to_csv(
            OUT_DIR / "wide_cv.csv", index=False)
    payload = {
        "combo_id": chosen["combo_id"], "combo_type": chosen["combo_type"],
        "groups": [g for g in str(chosen["groups"]).split("+")
                   if g and g not in ("nan", "None")],
        "n_features": len(cols), "criterion": criterion,
        "val_metrics": metrics, "cv_metrics": cv_rows,
        "caveat": "Dipilih di val; test tidak dipakai sama sekali di tahap ini.",
    }
    (OUT_DIR / "final_selection.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["features", "screen", "select",
                                            "wide", "select_wide", "all"], default="all")
    args = parser.parse_args()
    if args.stage in ("features", "all"):
        stage_features()
    if args.stage in ("screen", "all"):
        stage_screen()
    if args.stage in ("select", "all"):
        stage_select()
    if args.stage in ("wide", "all"):
        stage_wide()
    if args.stage in ("select_wide", "all"):
        stage_select_wide()
