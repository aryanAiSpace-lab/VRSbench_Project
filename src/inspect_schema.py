import json, argparse
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--json", required=True)
args = ap.parse_args()

with open(args.json) as f:
    data = json.load(f)

records = data if isinstance(data, list) else data.get("annotations", data.get("data"))
print(f"{len(records)} records")
print("Keys seen:", dict(Counter(k for r in records[:200] for k in r.keys())))
print("First record:", json.dumps(records[0], indent=2))