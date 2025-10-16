import csv
import json
import os

# === CONFIG ===
CSV_PATH = "arena_feedback.csv"            # TSV with model A/B, dimension, prompt, label
VBENCH_DIR = "vbench_jsons"                # Folder containing VBench dimension JSONs
OUTPUT_JSON = "vbench_reward_data.json"    # Where to save enriched reward dataset
DEFAULT_FRAMES = 16                        # Default frame count per video

# === STEP 1: Load arena_feedback.csv ===
print("[*] Loading arena feedback...")
data = []
with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    header = next(reader)  # skip header

    for row in reader:
        if len(row) != 5:
            continue  # skip malformed rows

        model_A, model_B, dimension, prompt, label = [r.strip() for r in row]

        if label not in ("0", "1"):
            continue  # skip unknown labels

        item = {
            "prompt": prompt,
            "path_A": f"{model_A}/{prompt}",
            "path_B": f"{model_B}/{prompt}",
            "num_frames_A": DEFAULT_FRAMES,
            "num_frames_B": DEFAULT_FRAMES,
            dimension: "A" if label == "1" else "B",
            f"MOS_A_{dimension}": 0.0,
            f"MOS_B_{dimension}": 0.0
        }
        data.append(item)

print(f"[✓] Loaded {len(data)} examples from {CSV_PATH}")

# === STEP 2: Load all VBench dimension JSONs ===
def load_vbench_jsons(vbench_dir):
    all_scores = {}
    for filename in os.listdir(vbench_dir):
        if filename.endswith(".json"):
            dim = filename.replace(".json", "")
            path = os.path.join(vbench_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                    all_scores[dim] = entries
                    print(f"[+] Loaded {len(entries)} scores from {filename}")
            except Exception as e:
                print(f"[!] Failed to load {filename}: {e}")
    return all_scores

print("[*] Loading VBench dimension scores...")
vbench_data = load_vbench_jsons(VBENCH_DIR)

# === STEP 3: Attach MOS scores from VBench to each example ===
print("[*] Attaching MOS scores to examples...")
for item in data:
    prompt = item["prompt"]
    model_A = item["path_A"].split("/")[0]
    model_B = item["path_B"].split("/")[0]

    for key in list(item.keys()):
        if key.startswith("MOS_A_"):
            dim = key.replace("MOS_A_", "")
            entries = vbench_data.get(dim, [])

            score_A = next((e["score"] for e in entries if e["model"] == model_A and e["prompt"] == prompt), 0.0)
            score_B = next((e["score"] for e in entries if e["model"] == model_B and e["prompt"] == prompt), 0.0)

            item[f"MOS_A_{dim}"] = score_A
            item[f"MOS_B_{dim}"] = score_B

# === STEP 4: Save result ===
print(f"[*] Saving output to {OUTPUT_JSON}...")
with open(OUTPUT_JSON, "w", encoding="utf-8") as out_file:
    json.dump(data, out_file, indent=2)
print(f"[✓] Done! Saved {len(data)} examples with MOS scores.")
