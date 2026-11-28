import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

df_fx = pd.read_csv("data/processed/trial5_filtered_llm_labels.csv")
df_fx["Tanggal_dt"] = pd.to_datetime(df_fx["Tanggal_dt"])

# Align cached roberta_title + roberta_fulltext (mean) to df_fx dates
unm = pd.read_csv("data/processed/news_processed_unmerged.csv", usecols=["aligned_trading_date"])
unm_dates = pd.to_datetime(unm["aligned_trading_date"]).dt.normalize()
rob_t = np.load("data/cache_embeddings/roberta_title.npy")
rob_b = np.load("data/cache_embeddings/roberta_fulltext.npy")
rob_art = 0.5 * rob_t + 0.5 * rob_b

date_to_idx = {d: i for i, d in enumerate(df_fx["Tanggal_dt"])}
X_rob = np.zeros((len(df_fx), 768), dtype=np.float32)
for dt_val, grp in unm.groupby(unm_dates):
    if dt_val in date_to_idx:
        X_rob[date_to_idx[dt_val]] = rob_art[grp.index].mean(axis=0)

n = len(df_fx)
n_tr, n_va = int(n * 0.70), int(n * 0.15)
idx_tr, idx_va, idx_te = np.arange(0, n_tr), np.arange(n_tr, n_tr+n_va), np.arange(n_tr+n_va, n)

y_dir = df_fx["label_dir_next"].values
y_tr, y_va, y_te = y_dir[idx_tr], y_dir[idx_va], y_dir[idx_te]

# Scale RoBERTa features
sc = StandardScaler().fit(X_rob[idx_tr])
Xtr_s, Xva_s, Xte_s = sc.transform(X_rob[idx_tr]), sc.transform(X_rob[idx_va]), sc.transform(X_rob[idx_te])

# 1. Lean MCM Head (768 -> 64 -> 2): Dropout -> Linear -> Tanh -> Dropout -> Linear
class LeanMCMHead(nn.Module):
    def __init__(self, in_dim=768, hidden=64, dropout=0.45):
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(hidden, 2)
        )
    def forward(self, x):
        return self.cls_head(x)

cnt0, cnt1 = (y_tr == 0).sum(), (y_tr == 1).sum()
cw = torch.tensor([len(y_tr)/(2.0*cnt0), len(y_tr)/(2.0*cnt1)], dtype=torch.float32)

torch.manual_seed(SEED)
mcm_head = LeanMCMHead(768, hidden=64, dropout=0.45)
opt = torch.optim.AdamW(mcm_head.parameters(), lr=1e-3, weight_decay=0.05)
crit = nn.CrossEntropyLoss(weight=cw)

Xt, Xv, Xte_t = torch.tensor(Xtr_s, dtype=torch.float32), torch.tensor(Xva_s, dtype=torch.float32), torch.tensor(Xte_s, dtype=torch.float32)
yt, yv = torch.tensor(y_tr, dtype=torch.long), torch.tensor(y_va, dtype=torch.long)

best_vl, best_w, wait = float("inf"), None, 0
for ep in range(100):
    mcm_head.train(); opt.zero_grad()
    loss = crit(mcm_head(Xt), yt)
    loss.backward(); opt.step()
    mcm_head.eval()
    with torch.no_grad():
        vl = crit(mcm_head(Xv), yv).item()
    if vl < best_vl - 1e-4:
        best_vl = vl; best_w = copy.deepcopy(mcm_head.state_dict()); wait = 0
    else:
        wait += 1
        if wait >= 15:
            print(f"LeanMCMHead stopped early at epoch {ep+1} (best val CE={best_vl:.4f})")
            break
mcm_head.load_state_dict(best_w); mcm_head.eval()
with torch.no_grad():
    p_mcm_va = torch.softmax(mcm_head(Xv), dim=-1)[:, 1].numpy()
    p_mcm_te = torch.softmax(mcm_head(Xte_t), dim=-1)[:, 1].numpy()

# 2. Frozen RoBERTa + PCA-32 + LogReg (Ultra-low param alternative!)
pca = PCA(n_components=32, random_state=SEED).fit(Xtr_s)
clf_pca = LogisticRegression(C=0.05, class_weight="balanced", random_state=SEED).fit(pca.transform(Xtr_s), y_tr)
p_pca_va = clf_pca.predict_proba(pca.transform(Xva_s))[:, 1]
p_pca_te = clf_pca.predict_proba(pca.transform(Xte_s))[:, 1]

# 3. LDA 8 topics
cv = CountVectorizer(max_features=1500, stop_words="english", min_df=3, max_df=0.85)
lda = LatentDirichletAllocation(n_components=8, learning_method="online", max_iter=15, random_state=SEED, n_jobs=1)
X_lda_tr = lda.fit_transform(cv.fit_transform(df_fx.iloc[idx_tr]["filtered_fx_text"]))
X_lda_va = lda.transform(cv.transform(df_fx.iloc[idx_va]["filtered_fx_text"]))
X_lda_te = lda.transform(cv.transform(df_fx.iloc[idx_te]["filtered_fx_text"]))
clf_lda = LogisticRegression(C=0.5, class_weight="balanced", random_state=SEED).fit(X_lda_tr, y_tr)
p_lda_va = clf_lda.predict_proba(X_lda_va)[:, 1]
p_lda_te = clf_lda.predict_proba(X_lda_te)[:, 1]

# 4. Hybrid Meta-Fusion (fit on Val)
X_m_va = np.hstack([p_mcm_va[:, None], p_pca_va[:, None], df_fx.iloc[idx_va][["llm_sentiment"]].values, X_lda_va])
X_m_te = np.hstack([p_mcm_te[:, None], p_pca_te[:, None], df_fx.iloc[idx_te][["llm_sentiment"]].values, X_lda_te])
sc_m = StandardScaler().fit(X_m_va)
meta = LogisticRegression(C=0.1, class_weight="balanced", random_state=SEED).fit(sc_m.transform(X_m_va), y_va)
p_hyb_te = meta.predict_proba(sc_m.transform(X_m_te))[:, 1]

def eval_c(name, prob):
    pred = (prob >= 0.5).astype(int)
    print(f"  {name:34s}: Acc={accuracy_score(y_te, pred)*100:.2f}% | BalAcc={balanced_accuracy_score(y_te, pred)*100:.2f}% | MacroF1={f1_score(y_te, pred, average='macro')*100:.2f}% | AUC={roc_auc_score(y_te, prob):.4f} | CM={confusion_matrix(y_te, pred).ravel()}")

print("\n=== RIGHT-SIZED TRIAL 5 RESULTS (TEST SET, N=147) ===")
eval_c("Baseline Mayoritas (Selalu 1)", np.ones_like(p_mcm_te))
eval_c("Topik LDA + LogReg (Balanced)", p_lda_te)
eval_c("Frozen RoBERTa + Lean MCM (768->64)", p_mcm_te)
eval_c("Frozen RoBERTa + PCA-32 + LogReg", p_pca_te)
eval_c("Hybrid Fusion (MCM+PCA+Sent+LDA)", p_hyb_te)
