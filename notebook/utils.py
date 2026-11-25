"""Helper bersama untuk notebook 02-05."""
import hashlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

RAW, PROC, CACHE = Path('../data/raw'), Path('../data/processed'), Path('../data/cache')
SEED = 42
K_TEXT = 6  # jumlah fitur teks yang dipilih (di dalam CV) untuk model return


def fingerprint(A):
    return hashlib.sha256('\n'.join(A.title + '|' + A.url).encode()).hexdigest()[:16]


def load(with_pup=True):
    """Satu tabel: target, split, fitur harga, fitur teks, (p_up). Hanya hari di dalam split."""
    d = pd.read_parquet(PROC / 'daily.parquet').set_index('date')
    r = d.ret
    P = pd.DataFrame({f'ret_lag{i}': r.shift(i) for i in range(5)})
    P['ret_roll_mean5'], P['ret_roll_std5'] = r.rolling(5).mean(), r.rolling(5).std()
    df = d[['target', 'split']].assign(target2=r.shift(-2), target0=r).join(P)
    df = df.join(pd.read_parquet(PROC / 'features_text.parquet').set_index('date'))
    if with_pup:
        df = df.join(pd.read_parquet(PROC / 'pup.parquet').set_index('date')['p_up'])
    return df[df.split != 'none'], list(P.columns)


def block(c):
    if c.startswith(('emT', 'emB', 'nov')): return 'emb'
    if c.startswith('top'): return 'topic'
    if c.startswith('gpr'): return 'gpr'
    if c.startswith('sent'): return 'sent'
    if c in ('news_count', 'has_news'): return 'cnt'
    return {'p_up': 'pup'}.get(c, 'price')


def variants(cols, price):
    """name -> (kolom yang selalu masuk, semua kolom). Fitur teks dipilih K_TEXT terbaik di dalam CV."""
    g = lambda *b: [c for c in cols if block(c) in b]
    return {'A': (price, price),
            'B': (price, price + g('cnt', 'gpr', 'sent')),
            'C': (price, price + g('cnt', 'gpr', 'sent', 'emb', 'topic')),
            'D': (price + ['p_up'], price + g('cnt', 'gpr', 'sent', 'emb', 'topic', 'pup'))}


def oof_pup(df, text_cols, C=0.05):
    """p_up: Train = OOF (TimeSeriesSplit, gap 5); Val = fit Train; Test = fit Train+Val."""
    y, X = (df.target > 0).astype(int), df[text_cols]
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))
    out = pd.Series(np.nan, index=df.index, name='p_up')
    tr = df.split == 'train'
    Xt, yt = X[tr], y[tr]
    for a, b in TimeSeriesSplit(5, gap=5).split(Xt):
        out[Xt.index[b]] = mk().fit(Xt.iloc[a], yt.iloc[a]).predict_proba(Xt.iloc[b])[:, 1]
    for s, fit_on in (('val', ['train']), ('test', ['train', 'val'])):
        te = df.split == s
        if te.any():
            f = df.split.isin(fit_on)
            out[te] = mk().fit(X[f], y[f]).predict_proba(X[te])[:, 1]
    return out


def one_se(cv, n=5):
    m = cv['mean_test_score']
    ok = np.where(m >= m.max() - cv['std_test_score'][m.argmax()] / np.sqrt(n))[0]
    return ok[np.argmax(np.asarray(cv['param_m__alpha'], float)[ok])]  # alpha terbesar = paling sederhana


def fit_predict(Xtr, ytr, Xte, keep, k=K_TEXT, rule='1se'):
    """Ridge + SelectKBest (di dalam CV), target distandardisasi dengan stat Train, aturan 1-SE."""
    text = [c for c in Xtr.columns if c not in keep]
    parts = [('k', 'passthrough', keep)] if keep else []
    if text: parts.append(('t', SelectKBest(f_regression, k=min(k, len(text))), text))
    pipe = Pipeline([('ct', ColumnTransformer(parts)), ('sc', StandardScaler()), ('m', Ridge())])
    gs = GridSearchCV(pipe, {'m__alpha': np.logspace(-1, 4, 11)}, cv=TimeSeriesSplit(5, gap=5),
                      scoring='neg_mean_squared_error', refit=one_se if rule == '1se' else True)
    mu, sd = ytr.mean(), ytr.std()
    gs.fit(Xtr, (ytr - mu) / sd)
    return gs.predict(Xte) * sd + mu


def evaluate(y, p, tr_std=None):
    out = dict(RMSE=np.sqrt(np.mean((y - p) ** 2)), MAE=np.mean(np.abs(y - p)), DirAcc=np.mean(np.sign(p) == np.sign(y)))
    if tr_std: out['collapsed'] = bool(np.std(p) < 0.05 * tr_std)  # prediksi ~konstan = "Prediksi 0"
    return out


def block_idx(n, block=5, rng=None):
    rng = rng or np.random.default_rng(SEED)
    s = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
    return (s[:, None] + np.arange(block)).ravel()[:n]


def auc_ci(y, p, B=1000, block=5):
    rng, out = np.random.default_rng(SEED), []
    for _ in range(B):
        i = block_idx(len(y), block, rng)
        if len(set(y[i])) == 2: out.append(roc_auc_score(y[i], p[i]))
    return roc_auc_score(y, p), np.percentile(out, [2.5, 97.5])


def dm_from_diff(d):
    """Diebold-Mariano dari selisih loss per hari d = loss1 - loss2 (h=1, koreksi Harvey). stat > 0: model 2 lebih baik."""
    d = np.asarray(d); n = len(d)
    s = d.mean() / np.sqrt(d.var(ddof=1) / n) * np.sqrt((n - 1) / n)
    return s, 2 * stats.t.sf(abs(s), n - 1)


def dm_test(y, p1, p2):
    return dm_from_diff((y - p1) ** 2 - (y - p2) ** 2)


def wf_blocks(n, start=60, step=5):
    """Walk-forward: latih ulang tiap `step` hari pada semua baris sebelumnya."""
    for i in range(start, n, step):
        yield np.arange(i), np.arange(i, min(i + step, n))


def walk_forward(df, specs, start=60, step=5, ycol='target', rule='min'):
    """specs: nama -> daftar kolom. Kembalikan prediksi out-of-sample + baseline rata-rata expanding."""
    out = {n: np.full(len(df), np.nan) for n in [*specs, 'mean']}
    for a, b in wf_blocks(len(df), start, step):
        tr, te = df.iloc[a], df.iloc[b]
        out['mean'][b] = tr[ycol].mean()
        for n, cols in specs.items():
            out[n][b] = fit_predict(tr[cols], tr[ycol], te[cols], cols, rule=rule)
    return pd.DataFrame(out, index=df.index).iloc[start:]


def rmse_diff_ci(y, p1, p2, B=1000, block=5):
    """Interval bootstrap blok untuk RMSE(model 1) - RMSE(model 2)."""
    rng, out = np.random.default_rng(SEED), []
    for _ in range(B):
        i = block_idx(len(y), block, rng)
        out.append(np.sqrt(np.mean((y[i] - p1[i]) ** 2)) - np.sqrt(np.mean((y[i] - p2[i]) ** 2)))
    return np.percentile(out, [2.5, 97.5])
