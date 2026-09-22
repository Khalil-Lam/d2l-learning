import argparse
import json
import random
import re
from pathlib import Path
from urllib.request import urlopen

BASE_URL = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/%E5%85%A8%E5%94%90%E8%AF%97/poet.tang.{idx}.json"


def clean_line(s: str) -> str:
    return re.sub(r"\s+", "", s.strip())


def valid_poem(item: dict) -> bool:
    title = clean_line(item.get("title", ""))
    paras = [clean_line(x) for x in item.get("paragraphs", [])]
    if not title or len(paras) < 2 or len(paras) > 8:
        return False
    text = "".join(paras)
    if len(text) < 16 or len(text) > 160:
        return False
    # Remove obvious annotations/rare editorial artifacts for a beginner-friendly subset.
    if any(ch in text for ch in "[]{}<>（）()《》"):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=3, help="Download poet.tang.0/1000/... files.")
    ap.add_argument("--limit", type=int, default=1200, help="Maximum number of poems to keep.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()

    rows = []
    for i in range(args.files):
        idx = i * 1000
        url = BASE_URL.format(idx=idx)
        print(f"Downloading {url}")
        with urlopen(url) as r:
            data = json.loads(r.read().decode("utf-8"))
        for item in data:
            if valid_poem(item):
                title = clean_line(item["title"])
                author = clean_line(item.get("author", "佚名"))
                poem = "\n".join(clean_line(x) for x in item["paragraphs"])
                rows.append({
                    "id": item.get("id", ""),
                    "title": title,
                    "author": author,
                    "prompt": f"请写一首题为《{title}》的唐诗：\n",
                    "completion": poem
                })

    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit]

    n = len(rows)
    n_train = int(n * 0.8)
    n_dev = int(n * 0.1)
    splits = {
        "train": rows[:n_train],
        "dev": rows[n_train:n_train+n_dev],
        "test": rows[n_train+n_dev:],
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        path = out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for x in items:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"{name}: {len(items)} -> {path}")


if __name__ == "__main__":
    main()
