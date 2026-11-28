"""
Script untuk membagi (split) dataset menjadi Train (70%), Validation (15%), dan Test (15%).

Script ini dirancang khusus untuk dataset NLP & Kurs BI (misal: data/processed/merged_data_clean.csv).
Mendukung 2 metode split:
1. 'chronological' (DEFAULT / DISARANKAN):
   - Data diurutkan berdasarkan tanggal dari terlama ke terbaru.
   - Train = 70% data awal, Val = 15% data tengah, Test = 15% data terbaru.
   - Mencegah lookahead bias / data leakage pada pemodelan time-series dan prediksi kurs.
2. 'random' (Shuffle):
   - Pembagian data secara acak menggunakan train_test_split scikit-learn.

Dapat dijalankan langsung melalui CLI atau di-import sebagai modul fungsi di Jupyter Notebook / script lain.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    df: pd.DataFrame,
    method: str = "chronological",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    date_col: str = "Tanggal_dt",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Membagi DataFrame menjadi 3 subset: train, val, dan test.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame yang akan dibagi.
    method : str
        'chronological' untuk split berurutan waktu (default), atau 'random' untuk split acak.
    train_ratio : float
        Proporsi data train (default: 0.70).
    val_ratio : float
        Proporsi data validation (default: 0.15).
    test_ratio : float
        Proporsi data test (default: 0.15).
    date_col : str
        Nama kolom tanggal untuk sorting jika menggunakan metode kronologis.
    random_state : int
        Seed angka acak untuk reproduktifitas jika menggunakan metode random.

    Returns:
    --------
    train_df, val_df, test_df : Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    # Validasi rasio
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.999 <= total_ratio <= 1.001):
        raise ValueError(
            f"Jumlah rasio harus sama dengan 1.0 (diterima: {train_ratio} + {val_ratio} + {test_ratio} = {total_ratio})"
        )

    n = len(df)
    if n == 0:
        raise ValueError("DataFrame kosong, tidak dapat melakukan split.")

    if method.lower() == "chronological":
        df_sorted = df.copy()
        if date_col in df_sorted.columns:
            # Pastikan format datetime dan urutan dari terlama (past) ke terbaru (present)
            df_sorted[date_col] = pd.to_datetime(df_sorted[date_col], errors="coerce")
            df_sorted = df_sorted.sort_values(date_col, ascending=True).reset_index(drop=True)
        else:
            print(f"[PERINGATAN] Kolom tanggal '{date_col}' tidak ditemukan. Menggunakan urutan baris saat ini.")
            df_sorted = df_sorted.reset_index(drop=True)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df_sorted.iloc[:train_end].copy().reset_index(drop=True)
        val_df = df_sorted.iloc[train_end:val_end].copy().reset_index(drop=True)
        test_df = df_sorted.iloc[val_end:].copy().reset_index(drop=True)

    elif method.lower() == "random":
        temp_ratio = val_ratio + test_ratio
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_ratio,
            random_state=random_state,
            shuffle=True,
        )

        val_relative_ratio = val_ratio / temp_ratio
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1.0 - val_relative_ratio),
            random_state=random_state,
            shuffle=True,
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
    else:
        raise ValueError(f"Metode '{method}' tidak dikenali. Pilih 'chronological' atau 'random'.")

    return train_df, val_df, test_df


def print_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str = "Tanggal_dt",
):
    """Mencetak ringkasan statistik hasil pembagian dataset."""
    total = len(train_df) + len(val_df) + len(test_df)
    print("\n" + "=" * 65)
    print(f"RINGKASAN HASIL SPLIT DATA (Total: {total:,} baris)")
    print("=" * 65)

    splits = [("Train", train_df), ("Validation", val_df), ("Test", test_df)]
    for name, split_df in splits:
        count = len(split_df)
        pct = (count / total) * 100

        extra_info = []
        if date_col in split_df.columns:
            dt_series = pd.to_datetime(split_df[date_col], errors="coerce").dropna()
            if not dt_series.empty:
                min_dt = dt_series.min().strftime("%Y-%m-%d")
                max_dt = dt_series.max().strftime("%Y-%m-%d")
                extra_info.append(f"Periode: {min_dt} s.d. {max_dt}")

        if "news_count" in split_df.columns:
            has_news = (split_df["news_count"] > 0).sum()
            extra_info.append(f"Hari ada berita: {has_news}/{count} ({has_news/count*100:.1f}%)")

        extra_str = (" | " + " | ".join(extra_info)) if extra_info else ""
        print(f"{name:12}: {count:5d} baris ({pct:5.1f}%){extra_str}")
    print("=" * 65 + "\n")


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str or Path,
    prefix: str = "",
) -> Tuple[Path, Path, Path]:
    """Menyimpan data train, val, dan test ke format CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prefix_str = f"{prefix}_" if prefix and not prefix.endswith("_") else prefix

    train_file = output_path / f"{prefix_str}train.csv"
    val_file = output_path / f"{prefix_str}val.csv"
    test_file = output_path / f"{prefix_str}test.csv"

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"[OK] File berhasil disimpan ke:")
    print(f"  - Train : {train_file}")
    print(f"  - Val   : {val_file}")
    print(f"  - Test  : {test_file}")

    return train_file, val_file, test_file


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset merged_data_clean.csv menjadi train (70%), val (15%), dan test (15%)."
    )
    parser.add_argument(
        "--input_path",
        "-i",
        type=str,
        default="data/processed/merged_data_clean.csv",
        help="Path ke file CSV input (default: data/processed/merged_data_clean.csv)",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="data/processed",
        help="Direktori output untuk menyimpan train.csv, val.csv, test.csv (default: data/processed)",
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        choices=["chronological", "random"],
        default="chronological",
        help="Metode pembagian data: 'chronological' (default, terurut waktu) atau 'random' (acak)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.70,
        help="Proporsi data train (default: 0.70)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Proporsi data val (default: 0.15)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Proporsi data test (default: 0.15)",
    )
    parser.add_argument(
        "--date_col",
        type=str,
        default="Tanggal_dt",
        help="Nama kolom tanggal untuk pengurutan kronologis (default: Tanggal_dt)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed untuk metode random (default: 42)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix opsional untuk nama file keluaran (misal: 'merged' -> merged_train.csv)",
    )

    args = parser.parse_args()

    input_file = Path(args.input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"File input tidak ditemukan di: {input_file.resolve()}")

    print(f"Membaca dataset dari: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Total baris dibaca: {len(df):,} baris, {len(df.columns)} kolom.")

    train_df, val_df, test_df = split_data(
        df=df,
        method=args.method,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        date_col=args.date_col,
        random_state=args.random_state,
    )

    print_split_summary(train_df, val_df, test_df, date_col=args.date_col)
    save_splits(train_df, val_df, test_df, output_dir=args.output_dir, prefix=args.prefix)


if __name__ == "__main__":
    main()
