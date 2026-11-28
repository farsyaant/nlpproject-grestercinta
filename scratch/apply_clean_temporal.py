import json

target_find = """text_daily_df["vol_5d"]   = text_daily_df["ret"].rolling(5).std()
text_daily_df["vol_20d"]  = text_daily_df["ret"].rolling(20).std()
text_daily_df["mom_5d"]   = text_daily_df["rate_clean"] / text_daily_df["rate_clean"].shift(5) - 1.0
text_daily_df["mom_20d"]  = text_daily_df["rate_clean"] / text_daily_df["rate_clean"].shift(20) - 1.0"""

replacement = """# PENTING: Fitur deret waktu di-lag/shift(1) agar 100% bebas dari lookahead bias (hanya menggunakan info s/d t-1)
text_daily_df["vol_5d"]   = text_daily_df["ret"].shift(1).rolling(5).std()
text_daily_df["vol_20d"]  = text_daily_df["ret"].shift(1).rolling(20).std()
text_daily_df["mom_5d"]   = text_daily_df["rate_clean"].shift(1) / text_daily_df["rate_clean"].shift(6) - 1.0
text_daily_df["mom_20d"]  = text_daily_df["rate_clean"].shift(1) / text_daily_df["rate_clean"].shift(21) - 1.0"""

# Update repo notebook directly
repo_path = "percobaan task 3/trial3_pipeline.ipynb"
with open(repo_path, "r", encoding="utf-8") as f:
    text = f.read()

# Normalize spacing if needed
if "vol_5d" in text:
    for line in text.splitlines():
        if 'vol_5d' in line:
            print("Found in repo:", line)
