"""Modular NLP feature-extraction & ablation-experiment utilities.

Dipisah sepenuhnya dari kode teman: tidak mengubah
`src/baseline_kurs_tengah.ipynb` maupun `src/preprocessing.ipynb`. File ini
hanya dipakai oleh eksperimen fitur NLP di
`notebook/nlp_feature_experiment.ipynb`.

Prinsip desain:
- `load_baseline_df_model()` mereproduksi persis langkah 1-5 di
  `baseline_kurs_tengah.ipynb` (Kurs Tengah, target NAIK/TURUN, fitur lag/
  rolling, dropna) supaya jumlah baris & urutan tanggalnya identik dengan
  baseline. Ini WAJIB supaya `chronological_split()` menghasilkan potongan
  train/val/test di tanggal yang sama persis dengan baseline (apple-to-apple).
- Semua fitur NLP yang butuh "belajar" dari data (TF-IDF + SVD) di-`fit`
  HANYA pada teks periode train lalu `.transform()` saja untuk val/test,
  supaya tidak ada data leakage. Fitur lexicon (VADER, Loughran-McDonald,
  keyword index) tidak perlu fitting karena kamusnya sudah tetap/offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pysentiment2 as ps
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# 1. Rekonstruksi df_model persis seperti baseline (untuk split yang identik)
# ---------------------------------------------------------------------------

TS_FEATURES = [
    "kurs_lag1", "kurs_lag2", "kurs_lag3", "kurs_lag4", "kurs_lag5",
    "kurs_roll_mean5", "kurs_roll_std5",
]


def load_baseline_df_model(raw_dir: Path) -> pd.DataFrame:
    """Reproduksi cell 3/5/7/9 di src/baseline_kurs_tengah.ipynb.

    Baris & urutan tanggal harus identik dengan baseline supaya split
    kronologis jatuh di tanggal yang sama persis.
    """
    bi_raw = pd.read_csv(raw_dir / "bi-usd-rate.csv")
    bi_raw["Tanggal_dt"] = pd.to_datetime(bi_raw["Tanggal"], errors="coerce")
    bi_raw = bi_raw.sort_values("Tanggal_dt").reset_index(drop=True)

    bi_raw["Kurs Tengah"] = (bi_raw["Kurs Jual"] + bi_raw["Kurs Beli"]) / 2
    bi_raw = bi_raw.drop(columns=["Kurs Jual", "Kurs Beli"])

    df_model = bi_raw.copy()
    df_model["kurs Next"] = df_model["Kurs Tengah"].shift(-1)
    df_model["Target"] = (df_model["kurs Next"] > df_model["Kurs Tengah"]).astype(int)
    df_model = df_model.iloc[:-1].copy()

    for lag in range(1, 6):
        df_model[f"kurs_lag{lag}"] = df_model["Kurs Tengah"].shift(lag)
    df_model["kurs_roll_mean5"] = df_model["Kurs Tengah"].shift(1).rolling(5).mean()
    df_model["kurs_roll_std5"] = df_model["Kurs Tengah"].shift(1).rolling(5).std()
    df_model["kurs_diff1"] = df_model["Kurs Tengah"] - df_model["kurs_lag1"]

    df_model = df_model.dropna(subset=TS_FEATURES + ["Target"]).reset_index(drop=True)
    return df_model


def chronological_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.85):
    """Split 70/15/15 berurutan waktu, sama persis dengan baseline cell 11."""
    n_rows = len(df)
    train_end = int(n_rows * train_frac)
    val_end = int(n_rows * val_frac)
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


# ---------------------------------------------------------------------------
# 2. Fitur NLP harian, dihitung di atas teks yang sudah diagregasi per hari
#    trading oleh tim (data/processed/merge_data.csv, hasil
#    src/preprocessing.ipynb).
# ---------------------------------------------------------------------------

def load_daily_news(processed_dir: Path) -> pd.DataFrame:
    news = pd.read_csv(processed_dir / "merge_data.csv")
    news["Tanggal_dt"] = pd.to_datetime(news["Tanggal_dt"], errors="coerce")
    news["all_titles"] = news["all_titles"].fillna("")
    news["all_text_lemmatized"] = news["all_text_lemmatized"].fillna("")
    # VADER pakai judul mentah (belum di-lemmatize) supaya negasi & tanda
    # baca tidak hilang -- keputusan yang sama seperti preprocessing.ipynb
    # cell 44 ("VADER -> lower_text"), diterapkan di level judul karena itu
    # satu-satunya teks non-lemmatized yang tersedia di file agregasi harian.
    news["daily_lower_text"] = news["all_titles"].str.lower()
    return news[["Tanggal_dt", "news_count", "daily_lower_text", "all_text_lemmatized"]]


_vader = SentimentIntensityAnalyzer()
_lm = ps.LM()  # kamus offline, tidak butuh koneksi internet

DEFAULT_GPR_KEYWORDS = [
    "geopolitics", "geopolitical risk", "geopolitical tensions",
    "geopolitical fragmentation", "armed conflict", "global conflict",
    "international conflict", "military tensions", "national security",
    "sanctions", "trade war", "tariffs", "embargo", "export controls",
    "protectionism", "supply chain disruption", "diplomacy",
    "foreign policy", "nato", "brics", "opec", "g7",
]
_GPR_PATTERN = re.compile("|".join(re.escape(k) for k in DEFAULT_GPR_KEYWORDS))

LEXICON_FEATURES = ["vader_compound", "lm_polarity", "lm_subjectivity", "gpr_density", "has_news"]


def add_lexicon_features(news: pd.DataFrame) -> pd.DataFrame:
    """Skenario A: VADER + Loughran-McDonald + indeks risiko geopolitik (5 kolom).

    Tidak butuh fitting -- semua berbasis kamus/aturan tetap, jadi aman
    dihitung sebelum split tanpa risiko leakage.
    """
    out = news.copy()

    vader_scores = out["daily_lower_text"].apply(_vader.polarity_scores).apply(pd.Series)
    out["vader_compound"] = vader_scores["compound"]

    def _lm_score(text: str) -> pd.Series:
        tokens = _lm.tokenize(text)
        s = _lm.get_score(tokens)
        return pd.Series({"lm_polarity": s["Polarity"], "lm_subjectivity": s["Subjectivity"]})

    lm_scores = out["all_text_lemmatized"].apply(_lm_score)
    out = pd.concat([out, lm_scores], axis=1)

    hits = out["daily_lower_text"].apply(lambda t: len(_GPR_PATTERN.findall(t)))
    word_count = out["daily_lower_text"].str.split().apply(len).replace(0, 1)
    out["gpr_density"] = hits / word_count

    out["has_news"] = (out["news_count"].fillna(0) > 0).astype(int)

    # hari tanpa berita -> netral (0), bukan NaN
    out[LEXICON_FEATURES] = out[LEXICON_FEATURES].fillna(0.0)
    return out


@dataclass
class TfidfSvdExtractor:
    """Vectorization klasik (TF-IDF + SVD/LSA). Wajib .fit() di teks TRAIN saja."""

    max_features: int = 5000
    n_components: int = 3
    ngram_range: tuple = (1, 2)
    min_df: int = 5
    random_state: int = 42

    _vectorizer: TfidfVectorizer = field(init=False, default=None, repr=False)
    _svd: TruncatedSVD = field(init=False, default=None, repr=False)

    def fit(self, texts: pd.Series) -> "TfidfSvdExtractor":
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,
        )
        tfidf_matrix = self._vectorizer.fit_transform(texts.fillna(""))
        n_comp = min(self.n_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        self._svd = TruncatedSVD(n_components=max(n_comp, 2), random_state=self.random_state)
        self._svd.fit(tfidf_matrix)
        return self

    def transform(self, texts: pd.Series) -> pd.DataFrame:
        if self._svd is None:
            raise RuntimeError("Panggil .fit(train_texts) dulu sebelum .transform().")
        tfidf_matrix = self._vectorizer.transform(texts.fillna(""))
        reduced = self._svd.transform(tfidf_matrix)
        cols = [f"tfidf_svd_{i}" for i in range(reduced.shape[1])]
        return pd.DataFrame(reduced, columns=cols, index=texts.index)

    @property
    def explained_variance_ratio_(self):
        return self._svd.explained_variance_ratio_


# ---------------------------------------------------------------------------
# 3. Model & evaluasi -- identik dengan baseline (cell 13/19) supaya adil
# ---------------------------------------------------------------------------

def evaluate_model(y_true, y_pred, model_name: str) -> dict:
    return {
        "model": model_name,
        "directional_accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def train_eval_logreg(feature_cols, train_df, val_df, test_df, model_name: str):
    """Logistic Regression + StandardScaler, di-fit HANYA di train (sama seperti baseline cell 19)."""
    X_train, y_train = train_df[feature_cols], train_df["Target"]
    X_val, y_val = val_df[feature_cols], val_df["Target"]
    X_test, y_test = test_df[feature_cols], test_df["Target"]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_s, y_train)

    rows = [
        evaluate_model(y_val, clf.predict(X_val_s), f"{model_name} - Val"),
        evaluate_model(y_test, clf.predict(X_test_s), f"{model_name} - Test"),
    ]
    return rows, clf
