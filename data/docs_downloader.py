from datasets import load_dataset
# ds = load_dataset("IlyaGusev/ru_news", split="train").shuffle(seed=7).select(range(5000))
# ds.save_to_disk("data/ru_news_5k")
from datasets import load_from_disk
import os, re

def slugify(s: str, default: str) -> str:
    s = re.sub(r"[\\s/\\\\:]+", "_", s.strip())
    s = re.sub(r"[^\\w\\-\\.]", "", s, flags=re.UNICODE)
    return s or default

ds = load_from_disk("data/ru_news_5k")
out_dir = "data/ru_news_txt"
os.makedirs(out_dir, exist_ok=True)

# попробуем угадать колонки
cols = ds.column_names
text_col = "text" if "text" in cols else cols[0]
title_col = "title" if "title" in cols else None

for i, row in enumerate(ds):
    title = (row.get(title_col) if title_col else None) or f"doc_{i:05d}"
    body = row.get(text_col) or ""
    fname = slugify(str(title)[:80], f"doc_{i:05d}") + ".txt"
    with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
        if title_col:
            f.write(str(title).strip() + "\n\n")
        f.write(str(body).strip())
print("Saved to", out_dir)
