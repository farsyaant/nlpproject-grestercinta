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
