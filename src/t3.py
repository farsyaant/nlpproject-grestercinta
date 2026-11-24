"""Helper Tugas 3: load export/cache, bangun fitur, pipeline, tuning. Notebook tinggal memanggil."""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config.json").read_text())
SEED = CFG["seed"]
EXPORT, OUT = ROOT / CFG["export_dir"], ROOT / CFG["out_dir"]
PRICE = [f"ret_lag{i}" for i in range(5)] + ["ret_roll_mean5", "ret_roll_std5"]  # fitur harga berbasis return

# ------------------------------------------------------------------ load
def _read(stem):
    pq = Path(stem).with_suffix(".parquet")
    df = pd.read_parquet(pq) if pq.exists() else pd.read_csv(Path(stem).with_suffix(".csv"))
    for c in ("date", "aligned_trading_date", "published_wib"):
        if c in df:
            df[c] = pd.to_datetime(df[c])
    return df


def fingerprint(articles):  # sama dengan make_embeddings.py
    return hashlib.sha1("\n".join(articles.sort_values("article_id")["url"].astype(str)).encode()).hexdigest()


def add_return_features(p):
    """ret_lag0..4 = return hari t..t-4 (dari ret + level kurs_lag1..5, jadi tidak ada baris yang hilang)."""
    L = np.log(p[[f"kurs_lag{i}" for i in range(1, 6)]].to_numpy())
    R = np.c_[p["ret"].to_numpy(), L[:, :-1] - L[:, 1:]]
    p = p.copy()
    p[[f"ret_lag{i}" for i in range(5)]] = R
    p["ret_roll_mean5"], p["ret_roll_std5"] = R.mean(1), R.std(1, ddof=1)
    return p


def load_export():
    prices, articles = add_return_features(_read(EXPORT / "prices_daily")), _read(EXPORT / "articles").sort_values("article_id").reset_index(drop=True)
    sent = _read(EXPORT / "sentiment_daily").sort_values("date").set_index("date")
    assert (articles["article_id"].to_numpy() == np.arange(len(articles))).all(), "article_id bukan 0..n-1"
    return prices, articles, sent


def _check(meta, articles, n):
    assert meta["articles_fingerprint"] == fingerprint(articles), "cache tidak cocok dengan tabel artikel"
    assert meta["n"] == n and meta["max_seq_length"] == CFG["max_seq_length"], "n / max_seq_length beda"


def load_emb(tag, articles):
    X = np.load(EXPORT / f"embeddings_{tag}.npy")
    _check(json.loads((EXPORT / f"embeddings_{tag}.json").read_text()), articles, len(X))
    return X, np.load(EXPORT / f"embeddings_{tag}_anchors.npy")


def load_finbert(articles):
    P = np.load(EXPORT / "finbert_probs.npy")
    meta = json.loads((EXPORT / "finbert_probs.json").read_text())
    _check(meta, articles, len(P))
    short = {"positive": "pos", "negative": "neg", "neutral": "neu"}
    return pd.DataFrame(P, columns=[short[l] for l in meta["labels"]])

# ------------------------------------------------------------------ fitur harian
def daily_finbert(articles, P, days):
    d = P.assign(score=P["pos"] - P["neg"], day=articles["aligned_trading_date"].values).groupby("day")
    out = pd.concat([d["score"].agg(fb_mean="mean", fb_min="min", fb_max="max", fb_std="std"),
                     d[["pos", "neg", "neu"]].mean().add_prefix("fb_")], axis=1)
    return out.reindex(days).fillna(0.0)


def daily_emb(articles, X, days):
    return pd.DataFrame(X).groupby(articles["aligned_trading_date"].values).mean().reindex(days).fillna(0.0)


def _unit(M):
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)


def build_features(prices, articles, sent, train_end):
    """Semua kolom fitur (index = baris prices) + dict blok -> kolom."""
    days = sent.index
    n = sent["news_count"]
    fb = daily_finbert(articles, load_finbert(articles), days)
    s = fb["fb_mean"]
    past = s.shift(1).rolling(10, min_periods=5)
    thr = fb.loc[(days <= train_end) & (n > 0), "fb_neg"].quantile(0.75)  # ambang dari train saja
    cal = {
        "sent_lama": sent.drop(columns="news_count"),
        "finbert": fb[["fb_mean", "fb_pos", "fb_neg", "fb_neu"]],
        "F1": pd.DataFrame({f"fb_lag{k}": s.shift(k) for k in (1, 2, 3)}),
        "F2": pd.DataFrame({f"fb_roll{k}": s.rolling(k, min_periods=1).mean() for k in (3, 5, 10)}),
        "F3": pd.DataFrame({"fb_diff": s.diff(), "fb_z": (s - past.mean()) / past.std().replace(0, np.nan)}),
        "F4": fb[["fb_min", "fb_max", "fb_std"]],
        "F5": pd.DataFrame({"news_count": n, "has_news": (n > 0).astype(int)}),
        "F8": pd.DataFrame({"regime_neg": (fb["fb_neg"] > thr).astype(int)}),
    }
    for tag in CFG["emb_tags"]:
        X, A = load_emb(tag, articles)
        E = _unit(daily_emb(articles, X, days).to_numpy())
        pm = pd.DataFrame(E).shift(1).rolling(5, min_periods=1).mean().fillna(0).to_numpy()
        cos = (E * _unit(pm)).sum(1)
        nov = np.where((np.linalg.norm(E, axis=1) > 0) & (np.linalg.norm(pm, axis=1) > 0), 1 - cos, 0.0)
        cal[f"emb_{tag}"] = pd.DataFrame(daily_emb(articles, X, days).to_numpy(), index=days).add_prefix(f"emb_{tag}_")
        cal[f"F7_{tag}"] = pd.DataFrame(np.c_[nov, E @ _unit(A).T], index=days,
                                         columns=["novelty"] + [f"anchor{i}" for i in range(len(A))]).add_suffix(f"_{tag}")
    cal = {k: v.reindex(prices["date"]).set_axis(prices.index).fillna(0.0) for k, v in cal.items()}
    sc = cal["finbert"]["fb_mean"]
    cal["F6"] = pd.DataFrame({"fb_x_std5": sc * prices["ret_roll_std5"], "fb_x_ret0": sc * prices["ret_lag0"]})
    blocks = {"harga": PRICE, **{k: list(v.columns) for k, v in cal.items()}}
    X = pd.concat([prices[["date", "split", "ret", "target", *PRICE]], *cal.values()], axis=1)
    return X, blocks

# ------------------------------------------------------------------ model & pipeline
MODELS = {  # nama: (pembuat model, search space, kompleksitas -- makin kecil makin sederhana)
    "Ridge": (lambda seed: Ridge(), {"alpha": loguniform(1e-1, 1e5)}, lambda p: -np.log(p["alpha"])),
    "ElasticNet": (lambda seed: ElasticNet(max_iter=5000),
                   {"alpha": loguniform(1e-4, 1), "l1_ratio": [0.1, 0.5, 0.9]}, lambda p: -np.log(p["alpha"])),
    "SVR": (lambda seed: SVR(), {"C": loguniform(1e-2, 10), "epsilon": loguniform(1e-2, 0.5)}, lambda p: p["C"]),
    "GradientBoosting": (lambda seed: GradientBoostingRegressor(random_state=seed),
                         {"n_estimators": [50, 100, 200], "max_depth": [1, 2, 3], "learning_rate": loguniform(1e-2, 0.2),
                          "subsample": [0.6, 0.8, 1.0], "min_samples_leaf": [5, 10, 20]},
                         lambda p: p["n_estimators"] * p["max_depth"]),
}


def make_pipe(cols, model):
    """Scaler (+ PCA untuk kolom emb_*) di dalam pipeline; target distandardisasi lalu di-inverse saat predict."""
    emb = [c for c in cols if c.startswith("emb_")]
    parts = [("num", StandardScaler(), [c for c in cols if c not in emb])]
    if emb:
        parts.append(("emb", make_pipeline(PCA(n_components=10, random_state=SEED), StandardScaler()), emb))
    return TransformedTargetRegressor(Pipeline([("prep", ColumnTransformer(parts)), ("model", model)]),
                                      transformer=StandardScaler())


class ResidualFusion(BaseEstimator, RegressorMixin):
    """Late fusion: model harga + model teks yang memprediksi sisa error OOF (expanding window + gap)."""
    def __init__(self, price, resid, n_splits=5, gap=5):
        self.price, self.resid, self.n_splits, self.gap = price, resid, n_splits, gap

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        oof = np.full(len(y), np.nan)
        for tr, te in TimeSeriesSplit(self.n_splits, gap=self.gap).split(X):
            oof[te] = clone(self.price).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        ok = np.where(~np.isnan(oof))[0]
        self.price_ = clone(self.price).fit(X, y)
        self.resid_ = clone(self.resid).fit(X.iloc[ok], (y - oof)[ok])
        return self

    def predict(self, X):
        return self.price_.predict(X) + self.resid_.predict(X)


def _cpx(params, f):
    k = next((v for n, v in params.items() if n.endswith("n_components")), 0)
    return k, f({n.split("__")[-1]: v for n, v in params.items()})


def _one_se(f):
    def refit(r):
        m = r["mean_test_score"]
        best = int(np.argmax(m))
        se = r["std_test_score"][best] / np.sqrt(CFG["n_splits"])
        return min(np.where(m >= m[best] - se)[0], key=lambda i: _cpx(r["params"][i], f))
    return refit


def tune(name, cols, X, y, late_price=None):
    """Random search (30 trial, TimeSeriesSplit expanding + gap, aturan 1-SE). Return dict hasil + model (3 seed utk GB)."""
    make, space, cpx = MODELS[name]
    pre = "resid__regressor__" if late_price is not None else "regressor__"

    def build(seed):
        est = make_pipe(cols, make(seed))
        return est if late_price is None else ResidualFusion(late_price, est, CFG["n_splits"], CFG["gap"])

    grid = {pre + "model__" + k: v for k, v in space.items()}
    if any(c.startswith("emb_") for c in cols):
        grid[pre + "prep__emb__pca__n_components"] = CFG["pca_k"]
    rs = RandomizedSearchCV(build(SEED), grid, n_iter=CFG["n_iter"], scoring=CFG["scoring"], refit=_one_se(cpx),
                            cv=TimeSeriesSplit(CFG["n_splits"], gap=CFG["gap"]), random_state=SEED, n_jobs=-1)
    rs.fit(X, y)
    i, r = rs.best_index_, rs.cv_results_
    seeds = [SEED + k for k in range(CFG["n_seeds"])] if name == "GradientBoosting" else [SEED]
    models = [clone(build(s)).set_params(**rs.best_params_).fit(X, y) for s in seeds]
    return {"cv_mse": -r["mean_test_score"][i], "cv_se": r["std_test_score"][i] / np.sqrt(CFG["n_splits"]),
            "params": rs.best_params_, "models": models}


def default_fit(name, cols, X, y, seed=SEED):
    return make_pipe(cols, MODELS[name][0](seed)).fit(X, y)


def predict(models, X):
    return np.mean([m.predict(X) for m in models], axis=0)


def block_importance(model, X, y, blocks, n_rep=20, seed=SEED):
    """Grouped permutation importance: kenaikan MSE saat satu blok kolom diacak (baris sama). Pakai data held-out."""
    rng = np.random.default_rng(seed)
    mse = lambda Z: np.mean((np.asarray(y) - model.predict(Z)) ** 2)
    base, out = mse(X), {}
    for b, cols in blocks.items():
        d = []
        for _ in range(n_rep):
            Z = X.copy()
            Z[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            d.append(mse(Z) - base)
        out[b] = (np.mean(d), np.std(d))
    return pd.DataFrame(out, index=["naik_MSE", "std"]).T.sort_values("naik_MSE", ascending=False)


def show(name):
    """Simpan figure ke outputs/figures lalu tampilkan."""
    import matplotlib.pyplot as plt
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT / "figures" / f"{name}.png", dpi=120)
    plt.show()
