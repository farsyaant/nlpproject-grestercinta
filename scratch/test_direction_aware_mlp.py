import re
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

df_raw = pd.read_csv('data/processed/merged_data_clean.csv',
                     usecols=['Tanggal_dt', 'Kurs Jual', 'Kurs Beli', 'all_title', 'all_full_text'])
fin_df = pd.read_csv('data/processed/finbert_unique_titles.csv')
fin_map = fin_df.set_index('title')[['finbert_pos', 'finbert_neg', 'finbert_neu']].to_dict('index')

df_raw['Tanggal_dt'] = pd.to_datetime(df_raw['Tanggal_dt'])
df_raw = df_raw.sort_values('Tanggal_dt').reset_index(drop=True)
df_raw['rate_clean'] = (df_raw['Kurs Jual'] + df_raw['Kurs Beli']) / 2.0

US_CASE = re.compile(r"\b(U\.S\.?|USA|US|USD|Fed|FOMC|Powell|Yellen|Bessent|Treasury|Wall Street|White House|Trump|Biden)\b")
US_LOWER = re.compile(r"\b(united states|america|american|federal reserve|the fed|dollar|greenback)\b")
IDR_CASE = re.compile(r"\b(IDR|BI|IHSG|Rupiah|Indonesia|Indonesian|Jakarta|Jokowi|Prabowo|Sri Mulyani|Perry Warjiyo|Bank Indonesia)\b")
IDR_LOWER = re.compile(r"\b(indonesia|indonesian|rupiah|jakarta|bank indonesia|bi rate|bi-rate|ihsg)\b")
GPR_SET = {
    "war", "military", "conflict", "geopolitic", "geopolitical", "threat", "sanction", "sanctions",
    "tension", "tensions", "crisis", "attack", "invasion", "nuclear", "terrorism", "terrorist",
    "missile", "hostility", "clash", "rebel", "insurgency", "troops", "sovereignty", "escalation",
    "retaliation", "dispute", "standoff", "blockade", "weapon", "drills", "border", "combat", "tariff", "tariffs"
}

daily_rows = []
for idx, row in df_raw.iterrows():
    titles = str(row['all_title']).split(' | ') if pd.notna(row['all_title']) else []
    bodies = str(row['all_full_text']).split(' | ') if pd.notna(row['all_full_text']) else []
    art_feats = []
    for i, t in enumerate(titles):
        t_c = t.strip()
        if not t_c or t_c.lower() == 'nan': continue
        b_c = bodies[i].strip() if i < len(bodies) else ""
        scores = fin_map.get(t_c, {'finbert_pos': 0.5, 'finbert_neg': 0.5})
        pos, neg = scores['finbert_pos'], scores['finbert_neg']
        pol_bin = 1.0 if pos >= neg else -1.0
        pol_margin = (pos - neg) / (pos + neg + 1e-8)
        
        full_t = f"{t_c} {b_c}"
        us_title = int(bool(US_CASE.search(t_c) or US_LOWER.search(t_c.lower())))
        idr_title = int(bool(IDR_CASE.search(t_c) or IDR_LOWER.search(t_c.lower())))
        us_cnt = len(US_CASE.findall(full_t)) + len(US_LOWER.findall(full_t.lower()))
        idr_cnt = len(IDR_CASE.findall(full_t)) + len(IDR_LOWER.findall(full_t.lower()))
        w_us = 1.0 if (us_title or us_cnt >= 2) else 0.0
        w_idr = 1.0 if (idr_title or idr_cnt >= 1) else 0.0
        
        toks = full_t.lower().split()
        gpr_d = sum(1 for w in toks if w in GPR_SET) / max(len(toks), 1)
        
        art_feats.append({
            'pol_bin': pol_bin, 'pol_margin': pol_margin,
            'flag_us': w_us, 'flag_idr': w_idr,
            'entity_dir_bin': w_us * pol_bin - w_idr * 2.0 * pol_bin,
            'entity_dir_margin': w_us * pol_margin - w_idr * 2.0 * pol_margin,
            'gpr_dens': gpr_d,
            'gpr_weak_bin': gpr_d * (-pol_bin),
            'gpr_weak_margin': gpr_d * (-pol_margin),
        })
    if art_feats:
        agg = pd.DataFrame(art_feats).mean().to_dict()
        agg['has_news'] = 1.0
    else:
        agg = {k: 0.0 for k in ['pol_bin', 'pol_margin', 'flag_us', 'flag_idr',
                                'entity_dir_bin', 'entity_dir_margin', 'gpr_dens',
                                'gpr_weak_bin', 'gpr_weak_margin']}
        agg['has_news'] = 0.0
    agg['date'] = row['Tanggal_dt']
    agg['rate_clean'] = row['rate_clean']
    daily_rows.append(agg)

daily = pd.DataFrame(daily_rows).set_index('date').sort_index()
daily['log_ret'] = np.log(daily['rate_clean'] / daily['rate_clean'].shift(1))
daily['target_dir'] = (daily['log_ret'] > 0).astype(int)

# Basic time-series (strictly t-1)
daily['ret_lag1'] = daily['log_ret'].shift(1)
daily['ret_lag2'] = daily['log_ret'].shift(2)
daily['vol_5d']   = daily['log_ret'].shift(1).rolling(5).std()
consecs, cur = [], 0
for r in daily['log_ret']:
    if pd.isna(r): consecs.append(0)
    elif r > 0: cur = cur + 1 if cur > 0 else 1; consecs.append(cur)
    elif r < 0: cur = cur - 1 if cur < 0 else -1; consecs.append(cur)
    else: cur = 0; consecs.append(0)
daily['consec_up_days'] = pd.Series(consecs, index=daily.index).shift(1)
ma20 = daily['rate_clean'].shift(1).rolling(20).mean()
daily['dist_ma20'] = daily['rate_clean'].shift(1) / ma20 - 1.0

base_txt_cols = ['has_news', 'pol_bin', 'pol_margin', 'flag_us', 'flag_idr',
                 'entity_dir_bin', 'entity_dir_margin', 'gpr_dens', 'gpr_weak_bin', 'gpr_weak_margin']
for c in base_txt_cols:
    daily[f'{c}_lag1'] = daily[c].shift(1)

valid = daily.dropna(subset=['log_ret', 'ret_lag2', 'vol_5d', 'consec_up_days', 'dist_ma20', 'pol_bin_lag1']).copy()
n = len(valid)
n_tr, n_va = int(n * 0.70), int(n * 0.15)
tr_m = np.arange(n) < n_tr
va_m = (np.arange(n) >= n_tr) & (np.arange(n) < n_tr + n_va)
te_m = np.arange(n) >= n_tr + n_va

# Stage 1 OOF with GBDT on Lagged Text (Entity+GPR+Pol)
s1_cols = [f"{c}_lag1" for c in base_txt_cols]
sc1 = StandardScaler().fit(valid.loc[tr_m, s1_cols])
X1_tr = sc1.transform(valid.loc[tr_m, s1_cols])
X1_va = sc1.transform(valid.loc[va_m, s1_cols])
X1_te = sc1.transform(valid.loc[te_m, s1_cols])
y_dir_tr = valid.loc[tr_m, 'target_dir'].values
y_dir_va = valid.loc[va_m, 'target_dir'].values
y_dir_te = valid.loc[te_m, 'target_dir'].values

kf = KFold(n_splits=5, shuffle=False)
oof_prob = np.zeros(tr_m.sum())
for f_tr, f_va in kf.split(X1_tr):
    c_f = GradientBoostingClassifier(n_estimators=40, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=SEED)
    c_f.fit(X1_tr[f_tr], y_dir_tr[f_tr])
    oof_prob[f_va] = c_f.predict_proba(X1_tr[f_va])[:, 1]

c_full = GradientBoostingClassifier(n_estimators=40, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=SEED).fit(X1_tr, y_dir_tr)
valid.loc[tr_m, 'prob_up_oof'] = oof_prob
valid.loc[va_m, 'prob_up_oof'] = c_full.predict_proba(X1_va)[:, 1]
valid.loc[te_m, 'prob_up_oof'] = c_full.predict_proba(X1_te)[:, 1]

# Stage 2 features
ts_cols = ['ret_lag1', 'ret_lag2', 'vol_5d', 'consec_up_days', 'dist_ma20']
s2_cols = ts_cols + ['prob_up_oof', 'entity_dir_margin_lag1', 'gpr_weak_margin_lag1']

sc2 = StandardScaler().fit(valid.loc[tr_m, s2_cols])
X2_tr = sc2.transform(valid.loc[tr_m, s2_cols])
X2_va = sc2.transform(valid.loc[va_m, s2_cols])
X2_te = sc2.transform(valid.loc[te_m, s2_cols])

y_tr = valid.loc[tr_m, 'log_ret'].values
y_va = valid.loc[va_m, 'log_ret'].values
y_te = valid.loc[te_m, 'log_ret'].values
mu_y, std_y = y_tr.mean(), y_tr.std()
y_tr_s = (y_tr - mu_y) / std_y
y_va_s = (y_va - mu_y) / std_y

class DirAwareMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_dir_mlp(dir_weight=0.0):
    torch.manual_seed(SEED)
    model = DirAwareMLP(X2_tr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    X_t = torch.tensor(X2_tr, dtype=torch.float32)
    y_t = torch.tensor(y_tr_s, dtype=torch.float32)
    y_raw_sign_t = torch.tensor(np.where(y_tr > 0, 1.0, -1.0), dtype=torch.float32)
    X_v = torch.tensor(X2_va, dtype=torch.float32)
    y_v = torch.tensor(y_va_s, dtype=torch.float32)
    y_raw_sign_v = torch.tensor(np.where(y_va > 0, 1.0, -1.0), dtype=torch.float32)
    
    best_score = float('inf')
    best_w = None
    for ep in range(200):
        model.train()
        opt.zero_grad()
        pred_s = model(X_t)
        # Convert back to raw log_ret scale inside graph to compute directional penalty around 0!
        pred_raw_scaled_to_0 = pred_s + (mu_y / std_y)
        mse = torch.mean((pred_s - y_t) ** 2)
        dir_loss = torch.mean(torch.nn.functional.softplus(-y_raw_sign_t * pred_raw_scaled_to_0 * 3.0))
        loss = mse + dir_weight * dir_loss
        loss.backward()
        opt.step()
        
        model.eval()
        with torch.no_grad():
            pv = model(X_v)
            pv_0 = pv + (mu_y / std_y)
            val_mse = torch.mean((pv - y_v) ** 2).item()
            val_dir = torch.mean(torch.nn.functional.softplus(-y_raw_sign_v * pv_0 * 3.0)).item()
            val_tot = val_mse + dir_weight * val_dir
        if val_tot < best_score:
            best_score = val_tot
            best_w = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_w)
    model.eval()
    with torch.no_grad():
        p_te = model(torch.tensor(X2_te, dtype=torch.float32)).numpy() * std_y + mu_y
    return p_te

def eval_m(name, y_p):
    err = y_te - y_p
    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    mse = np.mean(err**2)
    da = np.mean(np.sign(y_te) == np.sign(y_p)) * 100
    bacc = balanced_accuracy_score((y_te > 0).astype(int), (y_p > 0).astype(int)) * 100
    cm = confusion_matrix((y_te > 0).astype(int), (y_p > 0).astype(int)).ravel()
    print(f"  {name:32s}: RMSE={rmse:.6f} MAE={mae:.6f} DirAcc={da:.2f}% BalAcc={bacc:.2f}% CM={cm}")

print("=== DIRECTION-AWARE PYTORCH MLP BENCHMARK ===")
for dw in [0.0, 0.1, 0.2, 0.35]:
    p = train_dir_mlp(dir_weight=dw)
    eval_m(f"DirAwareMLP (dir_w={dw})", p)

# Also compare Ridge + DirAwareMLP blend
r100 = Ridge(alpha=100.0, random_state=SEED).fit(X2_tr, y_tr_s).predict(X2_te) * std_y + mu_y
p_dw = train_dir_mlp(dir_weight=0.2)
eval_m("Ridge (alpha=100)", r100)
eval_m("Blend (50% Ridge100 + 50% DirMLP)", 0.5 * r100 + 0.5 * p_dw)
