import json
from pathlib import Path

new_cell_8_source = [
    "# Ekstraksi Fitur Topik menggunakan BERTopic (Diperbaiki)\n",
    "cache_topic_path = Path(\"data/processed/bertopic_probabilities.csv\")\n",
    "\n",
    "tr_filter = df_raw[date_col] <= train_end\n",
    "train_docs = (df_raw.loc[tr_filter, \"title_clean\"] + \" \" + df_raw.loc[tr_filter, \"fulltext_clean\"]).tolist()\n",
    "all_docs   = (df_raw[\"title_clean\"] + \" \" + df_raw[\"fulltext_clean\"]).tolist()\n",
    "\n",
    "if cache_topic_path.exists():\n",
    "    print(f\"Memuat probabilitas BERTopic dari cache lokal: {cache_topic_path}\")\n",
    "    probs = pd.read_csv(cache_topic_path).values\n",
    "else:\n",
    "    print(\"Memulai ekstraksi fitur topik menggunakan BERTopic...\")\n",
    "    # Ganti dokumen kosong dengan placeholder agar UMAP/HDBSCAN stabil\n",
    "    train_docs_clean = [d if d.strip() else \"berita pasar uang ekonomi kurs valuta asing\" for d in train_docs]\n",
    "    all_docs_clean   = [d if d.strip() else \"berita pasar uang ekonomi kurs valuta asing\" for d in all_docs]\n",
    "    \n",
    "    try:\n",
    "        from bertopic import BERTopic\n",
    "        from sentence_transformers import SentenceTransformer\n",
    "        from umap import UMAP\n",
    "        from hdbscan import HDBSCAN\n",
    "        \n",
    "        # 1. Konfigurasi UMAP & HDBSCAN dengan seed deterministik\n",
    "        # (Catatan: random_state diatur pada UMAP, bukan di root BERTopic class)\n",
    "        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric=\"cosine\", random_state=SEED)\n",
    "        hdbscan_model = HDBSCAN(min_cluster_size=10, min_samples=5, metric=\"euclidean\", cluster_selection_method=\"eom\", prediction_data=True)\n",
    "        \n",
    "        # 2. Embedding model untuk representasi semantik kalimat\n",
    "        st_model = SentenceTransformer(\"all-MiniLM-L6-v2\", device=DEVICE)\n",
    "        \n",
    "        # 3. Inisialisasi BERTopic terstandar (dengan calculate_probabilities=True)\n",
    "        topic_model = BERTopic(\n",
    "            embedding_model=st_model,\n",
    "            umap_model=umap_model,\n",
    "            hdbscan_model=hdbscan_model,\n",
    "            nr_topics=5,\n",
    "            calculate_probabilities=True,\n",
    "            verbose=False\n",
    "        )\n",
    "        \n",
    "        # 4. Fit pada Train set saja (mencegah data leakage)\n",
    "        topic_model.fit(train_docs_clean)\n",
    "        \n",
    "        # 5. Transformasi seluruh dokumen ke 5 probabilitas topik\n",
    "        _, probs = topic_model.transform(all_docs_clean)\n",
    "        \n",
    "        if probs is None or len(probs.shape) == 1:\n",
    "            probs = np.zeros((len(all_docs), 5))\n",
    "        elif probs.shape[1] < 5:\n",
    "            probs = np.pad(probs, ((0, 0), (0, 5 - probs.shape[1])))\n",
    "        else:\n",
    "            probs = probs[:, :5]\n",
    "            \n",
    "        print(\"BERTopic berhasil mengekstrak 5 probabilitas topik!\")\n",
    "        # Simpan ke cache\n",
    "        cache_topic_path.parent.mkdir(parents=True, exist_ok=True)\n",
    "        pd.DataFrame(probs, columns=[f\"topic_prob_{k}\" for k in range(5)]).to_csv(cache_topic_path, index=False)\n",
    "        print(f\"Probabilitas topik berhasil di-cache ke disk: {cache_topic_path}\")\n",
    "        \n",
    "        # Tampilkan ringkasan topik\n",
    "        try:\n",
    "            topic_info = topic_model.get_topic_info()\n",
    "            print(\"\\nRingkasan Topik BERTopic:\")\n",
    "            display(topic_info.head(6))\n",
    "        except Exception:\n",
    "            pass\n",
    "            \n",
    "    except Exception as e:\n",
    "        print(f\"Fallback N-Gram Topic Model (karena: {e}):\")\n",
    "        from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "        from sklearn.decomposition import TruncatedSVD\n",
    "        \n",
    "        vec = TfidfVectorizer(max_features=500, stop_words=\"english\")\n",
    "        X_tr_tfidf = vec.fit_transform(train_docs_clean)\n",
    "        svd = TruncatedSVD(n_components=5, random_state=SEED)\n",
    "        svd.fit(X_tr_tfidf)\n",
    "        probs = svd.transform(vec.transform(all_docs_clean))\n",
    "        probs = np.clip(probs, 0, None)\n",
    "        print(\"N-Gram SVD Topics berhasil dibuat (5 komponen).\")\n",
    "\n",
    "for k in range(5):\n",
    "    df_raw[f\"topic_prob_{k}\"] = probs[:, k]\n",
    "\n",
    "print(\"Fitur topik berhasil ditambahkan ke dataset. Contoh (5 baris teratas):\")\n",
    "display(df_raw[[f\"topic_prob_{k}\" for k in range(5)]].head())\n"
]

def update_file(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    found = False
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and "topic_model = bertopic" in src.lower():
            print(f"Target found in cell {i} of {in_path}")
            cell["source"] = new_cell_8_source
            cell["outputs"] = []
            cell["execution_count"] = None
            found = True
            break
            
    if found:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Written to {out_path}")
    else:
        print(f"Not found in {in_path}")

update_file("percobaan task 3/trial3_pipeline.ipynb", "percobaan task 3/trial3_pipeline.ipynb")
update_file("d:/Downloads/trial-5.ipynb", "scratch/trial5_fixed.ipynb")
update_file("d:/Downloads/trial3_pipeline.ipynb", "scratch/trial3_fixed.ipynb")
