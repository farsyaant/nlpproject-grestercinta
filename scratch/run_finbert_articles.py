import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

t0 = time.time()
df = pd.read_csv('data/processed/merged_data_clean.csv')
articles = []
for idx, row in df.iterrows():
    titles = str(row['all_title']).split(' | ') if pd.notna(row['all_title']) else []
    for t in titles:
        t_clean = t.strip()
        if t_clean and t_clean.lower() != 'nan':
            articles.append(t_clean)

# Sort unique titles by length to minimize padding overhead on CPU
uniq_titles = sorted(list(set(articles)), key=len)
print(f"Total article occurrences: {len(articles)}, Unique titles: {len(uniq_titles)}", flush=True)

tok = AutoTokenizer.from_pretrained("ProsusAI/finbert", local_files_only=True)
mod = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", local_files_only=True)
mod.eval()
print(f"Loaded FinBERT in {time.time() - t0:.2f}s. Label mapping: {mod.config.id2label}", flush=True)

t1 = time.time()
pos_list, neg_list, neu_list = [], [], []
bs = 64
n_batches = (len(uniq_titles) + bs - 1) // bs

with torch.no_grad():
    for bi, i in enumerate(range(0, len(uniq_titles), bs)):
        b = uniq_titles[i:i+bs]
        inp = tok(b, padding=True, truncation=True, max_length=64, return_tensors='pt')
        probs = torch.softmax(mod(**inp).logits, dim=-1).numpy()
        pos_list.extend(probs[:, 0].tolist())
        neg_list.extend(probs[:, 1].tolist())
        neu_list.extend(probs[:, 2].tolist())
        if bi % 10 == 0 or bi == n_batches - 1:
            print(f"  Batch {bi+1}/{n_batches} (len~{inp['input_ids'].shape[1]}) elapsed {time.time() - t1:.1f}s", flush=True)

out_df = pd.DataFrame({
    'title': uniq_titles,
    'finbert_pos': pos_list,
    'finbert_neg': neg_list,
    'finbert_neu': neu_list,
})
out_df.to_csv('data/processed/finbert_unique_titles.csv', index=False)
print(f"Saved data/processed/finbert_unique_titles.csv in {time.time() - t1:.1f}s!", flush=True)
