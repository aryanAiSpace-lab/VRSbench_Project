from __future__ import annotations
import argparse, json, os, re
from collections import defaultdict
from typing import List, Dict, Any
from PIL import Image
from transformers import GPT2Tokenizer

CAPTION_TAG = "[caption]"
MIN_CAPTION_WORDS = 5


def load_train_records(json_path: str) -> List[Dict[str, str]]:
    """Train file is LLaVA-conversation-style and mixes captioning, VQA, and
    grounding together. Keep ONLY entries whose human turn contains the
    [caption] tag."""
    with open(json_path) as f:
        raw = json.load(f)
    out = []
    skipped_no_tag, skipped_bad_shape = 0, 0
    for rec in raw:
        convs = rec.get("conversations", [])
        if len(convs) < 2:
            skipped_bad_shape += 1
            continue
        human_turn = convs[0].get("value", "")
        gpt_turn = convs[1].get("value", "")
        if CAPTION_TAG not in human_turn:
            skipped_no_tag += 1
            continue
        out.append({"image": rec["image"], "caption": gpt_turn})
    print(f"  train: {len(out)} caption entries kept | {skipped_no_tag} non-caption skipped | {skipped_bad_shape} malformed skipped")
    return out


def load_eval_records(json_path: str) -> List[Dict[str, str]]:
    """Eval file is flat; defensively filter type=='caption' in case other
    task types are mixed into the same file."""
    with open(json_path) as f:
        raw = json.load(f)
    out = []
    skipped_wrong_type = 0
    for rec in raw:
        if rec.get("type") != "caption":
            skipped_wrong_type += 1
            continue
        out.append({"image": rec["image_id"], "caption": rec["ground_truth"]})
    print(f"  val: {len(out)} caption entries kept | {skipped_wrong_type} non-caption skipped")
    return out


def clean_caption(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text


def filter_bad_images(records, image_dir, bad_log_path):
    good, bad = [], []
    for rec in records:
        path = os.path.join(image_dir, str(rec["image"]))
        try:
            with Image.open(path) as img:
                img.verify()
            good.append(rec)
        except Exception as e:
            bad.append(f"{path}\t{e}")
    if bad:
        with open(bad_log_path, "a") as f:
            f.write("\n".join(bad) + "\n")
    print(f"  {len(good)} readable / {len(bad)} unreadable (logged to {bad_log_path})")
    return good


def clean_and_filter(records):
    cleaned = []
    for rec in records:
        cap = clean_caption(rec["caption"])
        if len(cap.split()) < MIN_CAPTION_WORDS:
            continue
        cleaned.append({"image": rec["image"], "caption": cap})
    print(f"  {len(cleaned)}/{len(records)} kept after degenerate-caption filter")
    return cleaned


def build_references(records):
    grouped = defaultdict(list)
    for rec in records:
        grouped[str(rec["image"])].append(rec["caption"])
    multi = {k: v for k, v in grouped.items() if len(v) > 1}
    print(f"  {len(multi)}/{len(grouped)} images have >1 caption")
    return grouped


def caption_length_stats(records, tokenizer):
    lengths = sorted(len(tokenizer(r["caption"])["input_ids"]) for r in records)
    n = len(lengths)
    pct = lambda p: lengths[int(n * p)]
    stats = {"n_captions": n, "mean_tokens": sum(lengths) / n,
             "p50_tokens": pct(0.50), "p90_tokens": pct(0.90),
             "p95_tokens": pct(0.95), "max_tokens": lengths[-1]}
    print(f"  mean={stats['mean_tokens']:.1f} p50={stats['p50_tokens']} "
          f"p95={stats['p95_tokens']} max={stats['max_tokens']}")
    return stats


def check_split_leakage(train_records, val_records):
    train_ids = {str(r["image"]) for r in train_records}
    val_ids = {str(r["image"]) for r in val_records}
    overlap = train_ids & val_ids
    print("  WARNING: overlap found!" if overlap else "  No train/val overlap.")
    return overlap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--val_json", required=True)
    ap.add_argument("--train_image_dir", required=True)
    ap.add_argument("--val_image_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--skip_image_check", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print("Loading + extracting captions from raw formats...")
    train_records = load_train_records(args.train_json)
    val_records = load_eval_records(args.val_json)

    print("\nChecking train/val split leakage...")
    check_split_leakage(train_records, val_records)

    if not args.skip_image_check:
        print("\nChecking train images...")
        train_records = filter_bad_images(train_records, args.train_image_dir, os.path.join(args.out_dir, "bad_images.txt"))
        print("Checking val images...")
        val_records = filter_bad_images(val_records, args.val_image_dir, os.path.join(args.out_dir, "bad_images.txt"))

    print("\nCleaning captions...")
    train_records = clean_and_filter(train_records)
    val_records = clean_and_filter(val_records)

    print("\nChecking for multiple references (val)...")
    val_references = build_references(val_records)

    print("\nCaption length stats (train)...")
    train_stats = caption_length_stats(train_records, tokenizer)

    json.dump(train_records, open(os.path.join(args.out_dir, "train_clean.json"), "w"))
    json.dump(val_records, open(os.path.join(args.out_dir, "val_clean.json"), "w"))
    json.dump(val_references, open(os.path.join(args.out_dir, "references.json"), "w"))
    json.dump({"train": train_stats, "n_train": len(train_records), "n_val": len(val_records)},
               open(os.path.join(args.out_dir, "stats.json"), "w"), indent=2)
    print(f"\nDone. Cleaned files in {args.out_dir}")


if __name__ == "__main__":
    main()