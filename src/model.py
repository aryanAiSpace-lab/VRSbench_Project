#1.Encoder (RemoteCLIP) — turns an image into a 512-number vector summarizing what's in it.
#2.Mapping network — a small neural net that translates that 512-number vector into a format GPT-2 understands (10 "pretend tokens" prepended before the real caption text).
#3.Decoder (GPT-2) — reads those 10 pretend image-tokens plus the real caption tokens, and learns to predict the next word — this is standard causal language modeling, just conditioned on the image.
import torch, torch.nn as nn, open_clip
from peft import LoraConfig, get_peft_model
from transformers import GPT2LMHeadModel

CLIP_DIM, GPT2_DIM = 512, 768

def build_encoder(name, ckpt_path=None):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=None if name == "remoteclip" else "openai")
    if name == "remoteclip":
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.visual, preprocess

def freeze(module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze_last_n_blocks(visual, n):
    freeze(visual)
    for block in visual.transformer.resblocks[-n:]:
        for p in block.parameters():
            p.requires_grad = True
    for p in visual.ln_post.parameters():
        p.requires_grad = True

class MappingNetwork(nn.Module):
    """The 'ClipCap' bridge — CLIP embedding -> prefix_len GPT-2-dim vectors."""
    def __init__(self, clip_dim, gpt2_dim, prefix_len):
        super().__init__()
        self.prefix_len, self.gpt2_dim = prefix_len, gpt2_dim
        self.net = nn.Sequential(
            nn.Linear(clip_dim, clip_dim * 2), nn.GELU(),
            nn.Linear(clip_dim * 2, prefix_len * gpt2_dim))

    def forward(self, clip_embed):
        return self.net(clip_embed).view(-1, self.prefix_len, self.gpt2_dim)

def build_decoder(use_lora, r=8, alpha=16, dropout=0.05):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    if use_lora:
        cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                          target_modules=["c_attn"], bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
    else:
        freeze(model)
    return model

class CaptionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.prefix_len = m["prefix_len"]
        self.visual, self.preprocess = build_encoder(m["encoder_name"], cfg["paths"].get("encoder_ckpt"))
        if m["freeze_encoder"]:
            freeze(self.visual)
        else:
            unfreeze_last_n_blocks(self.visual, m["encoder_unfreeze_last_n"])
        self.mapper = MappingNetwork(CLIP_DIM, GPT2_DIM, self.prefix_len)
        self.decoder = build_decoder(m["decoder_lora"], m["lora_r"], m["lora_alpha"], m["lora_dropout"])
        self.wte = self.decoder.get_input_embeddings()

    def forward(self, pixel_values, input_ids, attention_mask, labels):
        clip_embed = self.visual(pixel_values)
        prefix_embeds = self.mapper(clip_embed)
        token_embeds = self.wte(input_ids)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        return self.decoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(self, pixel_values, tokenizer, max_new_tokens=110):
        prefix_embeds = self.mapper(self.visual(pixel_values))
        out = self.decoder.generate(inputs_embeds=prefix_embeds, max_new_tokens=max_new_tokens,
                                     num_beams=4, repetition_penalty=1.2,
                                     pad_token_id=tokenizer.pad_token_id,
                                     eos_token_id=tokenizer.eos_token_id)
        return tokenizer.batch_decode(out, skip_special_tokens=True)