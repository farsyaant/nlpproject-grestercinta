import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Mari kita baca data yang sudah di-generate di test_user_idea_pipeline
exec(open('scratch/test_user_idea_pipeline.py').read())

print("\n--- EKSPERIMEN TUNING MODEL STAGE 2 DENGAN FITUR USER ---")
# Coba beberapa kombinasi model di Stage 2:
# 1. Ensemble Blend (MLP + Ridge)
pred_blend = 0.5 * pred_mlp_te + 0.5 * pred_ridge_te
eval_metrics(y_log_te, pred_blend, "Ensemble Blend (MLP+Ridge)")

# 2. Gradient Boosting
gbr = GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.03, random_state=42)
gbr.fit(X_s2_tr, y_log_tr_s)
pred_gbr_te = gbr.predict(X_s2_te) * std_log + mu_log
eval_metrics(y_log_te, pred_gbr_te, "Gradient Boosting (GBDT)")

# 3. Model Two-Stream: Arah dari Stage 1 Classifier dikalikan Magnitudo dari Stage 2 Regressor
# Arah: +1 jika prob_up >= 0.5, else -1
pred_dir_sign = np.where(pred_prob_te >= 0.5, 1.0, -1.0)
mag_tr = np.abs(y_log_tr)
sc_mag = StandardScaler().fit(valid.loc[tr_mask, ['vol_5d', 'gpr_density', 'dist_ma20']])
X_mag_tr = sc_mag.transform(valid.loc[tr_mask, ['vol_5d', 'gpr_density', 'dist_ma20']])
X_mag_te = sc_mag.transform(valid.loc[te_mask, ['vol_5d', 'gpr_density', 'dist_ma20']])
mag_model = Ridge(alpha=10.0).fit(X_mag_tr, mag_tr)
pred_mag = np.clip(mag_model.predict(X_mag_te), 0.0005, 0.01)
pred_hybrid = pred_dir_sign * pred_mag
eval_metrics(y_log_te, pred_hybrid, "Hybrid (Stage1 Sign * Mag)")

# Cek koefisien Ridge untuk melihat pengaruh masing-masing fitur
coef_df = pd.Series(ridge_s2.coef_, index=stage2_features)
print("\nKoefisien Ridge Regressor (Feature Importance):")
print(coef_df.round(4))
