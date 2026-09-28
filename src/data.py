import json, os, torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

KEY_IMAGE = "image"        # matches preprocess.py's normalized output
KEY_CAPTION = "caption"
IGNORE_INDEX = -100

def build_tokenizer():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token   # GPT-2 has no pad token by default
    return tok

class VRSBenchDataset(Dataset):
    def __init__(self, json_path, image_dir, preprocess, tokenizer, max_len=110, subset=None):
        with open(json_path) as f:
            self.records = json.load(f)   # already clean list from preprocess.py
        if subset:
            self.records = self.records[:subset]
        self.image_dir, self.preprocess, self.tokenizer, self.max_len = image_dir, preprocess, tokenizer, max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(os.path.join(self.image_dir, str(rec[KEY_IMAGE]))).convert("RGB")
        pixel_values = self.preprocess(image)
        enc = self.tokenizer(rec[KEY_CAPTION], max_length=self.max_len,
                              truncation=True, padding="max_length", return_tensors="pt")
        return {"pixel_values": pixel_values,
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "caption": rec[KEY_CAPTION],
                "image": str(rec[KEY_IMAGE])}   # needed later for multi-ref lookup at eval time

class CollateFn:
    """A class instead of a closure — Windows' multiprocessing DataLoader
    workers need to pickle the collate function, and a nested function
    (closure) can't be pickled on Windows. A class with plain attributes can."""
    def __init__(self, prefix_len):
        self.prefix_len = prefix_len

    def __call__(self, batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        labels = input_ids.clone()
        labels[attention_mask == 0] = IGNORE_INDEX
        prefix_labels = torch.full((input_ids.size(0), self.prefix_len), IGNORE_INDEX, dtype=torch.long)
        labels = torch.cat([prefix_labels, labels], dim=1)
        prefix_attn = torch.ones((input_ids.size(0), self.prefix_len), dtype=attention_mask.dtype)
        attention_mask = torch.cat([prefix_attn, attention_mask], dim=1)
        return {"pixel_values": pixel_values, "input_ids": input_ids,
                "attention_mask": attention_mask, "labels": labels,
                "captions": [b["caption"] for b in batch],
                "images": [b["image"] for b in batch]}

def make_collate_fn(prefix_len):
    return CollateFn(prefix_len)