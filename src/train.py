import argparse, os, torch
from torch.utils.data import DataLoader
from config import load_config
from data import VRSBenchDataset, build_tokenizer, make_collate_fn
from model import CaptionModel
from utils import save_checkpoint, load_checkpoint, gpu_mem_mb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--mode", required=True)
    args = ap.parse_args()

    cfg = load_config(args.platform, args.mode)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = build_tokenizer()
    model = CaptionModel(cfg).to(device)

    if cfg["train"]["grad_checkpointing"]:
        model.decoder.gradient_checkpointing_enable()
        model.decoder.config.use_cache = False

    dataset = VRSBenchDataset(cfg["paths"]["train_json"], cfg["paths"]["train_image_dir"],
                               model.preprocess, tokenizer,
                               max_len=cfg["data"]["max_caption_len"],
                               subset=cfg["data"]["train_subset"])
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=True,
                         num_workers=cfg["train"]["num_workers"],
                         collate_fn=make_collate_fn(cfg["model"]["prefix_len"]))

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["train"]["lr"])
    use_amp = cfg["train"]["precision"] == "fp16" and device == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    output_dir = cfg["paths"]["output_dir"]
    resume_path = os.path.join(output_dir, "last.pt")
    step, epoch = 0, 0
    if os.path.exists(resume_path):
        step, epoch = load_checkpoint(resume_path, model, optimizer, scaler, device)
        print(f"Resumed at step {step}")

    accum = cfg["train"]["grad_accum_steps"]
    model.train()
    for e in range(epoch, cfg["train"]["epochs"]):
        optimizer.zero_grad()
        for i, batch in enumerate(loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                out = model(pixel_values=batch["pixel_values"], input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"], labels=batch["labels"])
                loss = out.loss / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            step += 1
            if step % cfg["train"]["log_every_steps"] == 0:
                print(f"step {step} loss {out.loss.item():.4f} gpu_mem {gpu_mem_mb()}MB")
            if step % cfg["train"]["save_every_steps"] == 0:
                save_checkpoint(resume_path, model, optimizer, scaler, step, e)
    save_checkpoint(resume_path, model, optimizer, scaler, step, cfg["train"]["epochs"] - 1)

if __name__ == "__main__":
    main()