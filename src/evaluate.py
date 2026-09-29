import argparse, csv, json, os, torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from bert_score import score as bert_score
from config import load_config
from data import VRSBenchDataset, build_tokenizer, make_collate_fn
from model import CaptionModel
from utils import load_checkpoint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    cfg = load_config(args.platform, args.mode)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = build_tokenizer()
    model = CaptionModel(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    dataset = VRSBenchDataset(cfg["paths"]["val_json"], cfg["paths"]["val_image_dir"],
                               model.preprocess, tokenizer,
                               max_len=cfg["data"]["max_caption_len"],
                               subset=cfg["data"]["val_subset"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg["train"]["batch_size"],
                                          collate_fn=make_collate_fn(cfg["model"]["prefix_len"]))

    refs_path = cfg["paths"].get("references_json")
    multi_refs = json.load(open(refs_path)) if refs_path and os.path.exists(refs_path) else None

    image_names, single_refs, hyps = [], [], []
    from tqdm import tqdm

    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating captions"):
            preds = model.generate(batch["pixel_values"].to(device), tokenizer)
            hyps.extend(preds)
            single_refs.extend(batch["captions"])
            image_names.extend(batch["images"])    
    if multi_refs:
        bleu_refs = [[r.split() for r in multi_refs.get(img, [cap])] for img, cap in zip(image_names, single_refs)]
    else:
        bleu_refs = [[c.split()] for c in single_refs]

    smoothie = SmoothingFunction().method4
    bleu4 = corpus_bleu(bleu_refs, [h.split() for h in hyps],
                         weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    _, _, F1 = bert_score(hyps, single_refs, lang="en")
    print(f"BLEU-4: {bleu4:.4f} | BERTScore F1: {F1.mean().item():.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/summary.csv", "a", newline="") as f:
        csv.writer(f).writerow([cfg["run_name"], args.checkpoint, bleu4, F1.mean().item()])

if __name__ == "__main__":
    main()