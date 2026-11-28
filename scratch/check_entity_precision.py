import re
import pandas as pd

df = pd.read_csv('data/processed/merged_data_clean.csv')

# Case-sensitive abbreviations + case-insensitive full names for US vs IDR
US_REGEX_CASE = r"\b(U\.S\.?|USA|US|USD|Fed|Powell|Yellen|Bessent|Wall Street|Treasury|White House|Trump|Biden)\b"
US_REGEX_LOWER = r"\b(united states|america|american|federal reserve|dollar|greenback|u\.s\.|usa)\b"

IDR_REGEX_CASE = r"\b(IDR|BI|IHSG|Rupiah|Indonesia|Indonesian|Jakarta|Jokowi|Prabowo|Sri Mulyani|Perry Warjiyo|Bank Indonesia)\b"
IDR_REGEX_LOWER = r"\b(indonesia|indonesian|rupiah|jakarta|bank indonesia|bi rate|bi-rate|ihsg|sri mulyani|perry warjiyo)\b"

total_art = 0
us_hits = 0
idr_hits = 0
both_hits = 0
neither_hits = 0
sample_idr = []
sample_us = []

for idx, row in df.iterrows():
    titles = str(row['all_title']).split(' | ') if pd.notna(row['all_title']) else []
    bodies = str(row['all_full_text']).split(' | ') if pd.notna(row['all_full_text']) else []
    for i, t in enumerate(titles):
        b = bodies[i] if i < len(bodies) else ""
        # Check title first, and also full text
        raw_text = f"{t} {b}"
        low_text = raw_text.lower()
        
        is_us = bool(re.search(US_REGEX_CASE, raw_text)) or bool(re.search(US_REGEX_LOWER, low_text))
        is_idr = bool(re.search(IDR_REGEX_CASE, raw_text)) or bool(re.search(IDR_REGEX_LOWER, low_text))
        
        # Also title-only check
        t_us = bool(re.search(US_REGEX_CASE, t)) or bool(re.search(US_REGEX_LOWER, t.lower()))
        t_idr = bool(re.search(IDR_REGEX_CASE, t)) or bool(re.search(IDR_REGEX_LOWER, t.lower()))
        
        total_art += 1
        if is_us and is_idr: both_hits += 1
        elif is_us: us_hits += 1
        elif is_idr: idr_hits += 1
        else: neither_hits += 1
        
        if t_idr and len(sample_idr) < 5:
            sample_idr.append(t[:100])
        if t_us and not t_idr and len(sample_us) < 5:
            sample_us.append(t[:100])

print(f"Total articles: {total_art}")
print(f"Full text flags -> US only: {us_hits} ({us_hits/total_art*100:.1f}%), IDR only: {idr_hits} ({idr_hits/total_art*100:.1f}%), Both: {both_hits} ({both_hits/total_art*100:.1f}%), Neither: {neither_hits} ({neither_hits/total_art*100:.1f}%)")
print("Sample IDR titles:", sample_idr)
print("Sample US titles:", sample_us)
