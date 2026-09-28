import yaml, copy, os

def deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged

def load_config(platform, mode):
    config_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
    with open(os.path.join(config_dir, f"{platform}.yaml")) as f:
        platform_cfg = yaml.safe_load(f)
    with open(os.path.join(config_dir, f"mode_{mode}.yaml")) as f:
        mode_cfg = yaml.safe_load(f)
    cfg = deep_merge(platform_cfg, mode_cfg)
    cfg["run_name"] = f"{platform}_{mode}"
    cfg["paths"]["output_dir"] = os.path.join(cfg["paths"]["output_root"], cfg["run_name"])
    return cfg