from huggingface_hub import hf_hub_download
import shutil, os, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="checkpoints/RemoteCLIP-ViT-B-32.pt")
args = ap.parse_args()

path = hf_hub_download(repo_id="chendelong/RemoteCLIP", filename="RemoteCLIP-ViT-B-32.pt",
                        cache_dir="checkpoints/.hf_cache")
os.makedirs(os.path.dirname(args.out), exist_ok=True)
shutil.copy(path, args.out)
print(f"Saved to {args.out}")