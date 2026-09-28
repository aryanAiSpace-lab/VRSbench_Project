import torch, os

def save_checkpoint(path, model, optimizer, scaler, step, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler else None,
        "step": step, "epoch": epoch,
    }, tmp_path)
    os.replace(tmp_path, path)

def load_checkpoint(path, model, optimizer=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and ckpt.get("optimizer_state"):
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler and ckpt.get("scaler_state"):
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt["step"], ckpt["epoch"]

def gpu_mem_mb():
    return torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None