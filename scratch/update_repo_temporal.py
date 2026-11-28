import json

def update_nb_clean_temporal(p):
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            src = cell.get("source", [])
            new_src = []
            for line in src:
                if 'text_daily_df["vol_5d"]' in line and '.shift(1)' not in line:
                    new_src.append('text_daily_df["vol_5d"]   = text_daily_df["ret"].shift(1).rolling(5).std()\n')
                    modified = True
                elif 'text_daily_df["vol_20d"]' in line and '.shift(1)' not in line:
                    new_src.append('text_daily_df["vol_20d"]  = text_daily_df["ret"].shift(1).rolling(20).std()\n')
                    modified = True
                elif 'text_daily_df["mom_5d"]' in line and '.shift(1)' not in line:
                    new_src.append('text_daily_df["mom_5d"]   = text_daily_df["rate_clean"].shift(1) / text_daily_df["rate_clean"].shift(6) - 1.0\n')
                    modified = True
                elif 'text_daily_df["mom_20d"]' in line and '.shift(1)' not in line:
                    new_src.append('text_daily_df["mom_20d"]  = text_daily_df["rate_clean"].shift(1) / text_daily_df["rate_clean"].shift(21) - 1.0\n')
                    modified = True
                else:
                    new_src.append(line)
            cell["source"] = new_src
            
    if modified:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Updated {p}")
    else:
        print(f"No changes needed for {p}")

update_nb_clean_temporal("percobaan task 3/trial3_pipeline.ipynb")
