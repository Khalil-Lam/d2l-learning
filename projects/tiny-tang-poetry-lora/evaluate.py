import argparse
import json
import re
from pathlib import Path

from generate import generate


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def normalize(s):
    return re.sub(r"\s+", "", s)


def line_lengths(text):
    parts = [x.strip() for x in re.split(r"[。！？!?]", text) if x.strip()]
    lengths = []
    for p in parts:
        p = re.sub(r"[，、；：,.!?！？。；：]", "", p)
        if p:
            lengths.append(len(p))
    return lengths


def stats(text, train_poems):
    n = normalize(text)
    lens = line_lengths(text)
    regular = sum(x in (5, 7, 10, 14) for x in lens) / len(lens) if lens else 0.0
    exact_train_match = n in train_poems
    return {
        "chars": len(n),
        "segments": len(lens),
        "regular_segment_rate": round(regular, 3),
        "exact_train_match": exact_train_match,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="outputs/tang-poetry-lora")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="runs/comparison.json")
    args = ap.parse_args()

    test = read_jsonl(args.test)[: args.n]
    train = read_jsonl(args.train)
    train_poems = {normalize(x["completion"]) for x in train}

    rows = []
    for i, item in enumerate(test):
        title = item["title"]
        print(f"[{i+1}/{len(test)}] {title}")
        base = generate(title, adapter=None, seed=42+i)
        tuned = generate(title, adapter=args.adapter, seed=42+i)
        rows.append({
            "title": title,
            "reference": item["completion"],
            "baseline": base,
            "lora": tuned,
            "baseline_stats": stats(base, train_poems),
            "lora_stats": stats(tuned, train_poems),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved comparison to {out}")


if __name__ == "__main__":
    main()
