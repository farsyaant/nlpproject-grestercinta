"""Metrik evaluasi bersama untuk prediksi log return kurs USD/IDR.

Semua anggota tim sebaiknya memakai fungsi di sini (jangan menulis ulang),
supaya MAE / MSE / RMSE / DirAcc / PredStd dihitung dengan cara yang sama.

Definisi:
- MAE, MSE, RMSE : error standar antara target dan prediksi (satuan: log return).
- DirAcc         : proporsi hari yang arahnya (tanda +/-) tertebak benar.
                   Hari dengan target tepat 0 tidak dihitung (arahnya tidak ada).
                   Prediksi tepat 0 dianggap salah arah (sign 0 != sign +/-1).
- PredStd        : standar deviasi prediksi (ddof=0). Mendekati 0 berarti model
                   menebak angka yang hampir sama terus.
"""
from __future__ import annotations

import numpy as np

METRICS = ["MAE", "MSE", "RMSE", "DirAcc"]  # metrik utama untuk tabel perbandingan
ALL_METRICS = METRICS + ["PredStd"]         # semua yang dikembalikan evaluate()


def evaluate(y_true, y_pred) -> dict:
    """Hitung MAE, MSE, RMSE, DirAcc, dan PredStd untuk satu set prediksi."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Panjang beda: y_true={y_true.shape}, y_pred={y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("y_true kosong")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("y_true / y_pred mengandung NaN atau inf")

    err = y_pred - y_true
    mse = float(np.mean(err ** 2))

    nonzero = y_true != 0
    if nonzero.any():
        dir_acc = float(np.mean(np.sign(y_pred[nonzero]) == np.sign(y_true[nonzero])))
    else:
        dir_acc = float("nan")

    return {
        "MAE": float(np.mean(np.abs(err))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "DirAcc": dir_acc,
        "PredStd": float(np.std(y_pred)),
    }


def naive_baselines(y_true, ret_today) -> dict:
    """Pembanding tanpa model: tebak 0, atau tebak return besok = return hari ini."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    return {
        "Prediksi 0": evaluate(y_true, np.zeros(len(y_true))),
        "Return kemarin": evaluate(y_true, ret_today),
    }


# ---------------------------------------------------------------------------
# Tambahan Tugas 3. `evaluate` dan `naive_baselines` di atas TIDAK diubah.
# ---------------------------------------------------------------------------
def up_share(y_true) -> dict:
    """Proporsi hari naik / turun / nol pada target."""
    y = np.asarray(y_true, dtype=float).ravel()
    nz = y[y != 0]
    return {"naik": float((nz > 0).mean()), "turun": float((nz < 0).mean()), "nol": float((y == 0).mean())}


def majority_baselines(y_true) -> dict:
    """DirAcc untuk "selalu tebak naik" / "selalu tebak turun" (target 0 dilewati).
    Model baru dianggap punya sinyal arah kalau DirAcc-nya melampaui "Mayoritas"."""
    s = up_share(y_true)
    return {"Selalu naik": s["naik"], "Selalu turun": s["turun"], "Mayoritas": max(s["naik"], s["turun"])}


def is_collapsed(y_pred, y_true, tol: float = 0.05) -> bool:
    """True kalau std prediksi < tol * std target (model menebak hampir konstan)."""
    return bool(np.std(y_pred) < tol * np.std(y_true))


def dm_test(y_true, pred_a, pred_b, loss: str = "se", h: int = 1):
    """Diebold-Mariano (koreksi Harvey, t-dist). Return (stat, p).
    stat < 0 berarti A lebih akurat dari B. loss: "se" (kuadrat) atau "ae" (absolut)."""
    from scipy import stats
    y, a, b = (np.asarray(v, dtype=float).ravel() for v in (y_true, pred_a, pred_b))
    ea, eb = y - a, y - b
    d = ea ** 2 - eb ** 2 if loss == "se" else np.abs(ea) - np.abs(eb)
    n = len(d)
    dc = d - d.mean()
    gamma = [(dc[k:] * dc[: n - k]).sum() / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return float("nan"), float("nan")
    dm = d.mean() / np.sqrt(var) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(dm), float(2 * stats.t.sf(abs(dm), n - 1))


def block_bootstrap_ci(y_true, y_pred, metric: str = "RMSE", ref_pred=None,
                       block: int = 10, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    """CI moving-block bootstrap untuk metrik. Kalau `ref_pred` diberikan, CI untuk
    selisih metrik(y_pred) - metrik(ref_pred); negatif = y_pred lebih baik."""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    r = None if ref_pred is None else np.asarray(ref_pred, dtype=float).ravel()
    n = len(y)
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    vals = []
    for i in idx:
        v = evaluate(y[i], p[i])[metric]
        if r is not None:
            v -= evaluate(y[i], r[i])[metric]
        vals.append(v)
    lo, hi = np.nanquantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
