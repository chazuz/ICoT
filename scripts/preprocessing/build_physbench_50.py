import json
import random
import os
from collections import defaultdict
from PIL import Image

INPUT_PATH = "data/processed/physbench_full.json"
OUTPUT_PATH = "data/processed/physbench_50_seed0.json"

SEED = 0
random.seed(SEED)

VALID_ANSWERS = {"A", "B", "C", "D"}

# -----------------------------
# UPDATED DISTRIBUTION (50% dynamics)
# -----------------------------
TARGET_DISTRIBUTION = {
    "dynamics": 25,        # 50%
    "relationships": 15,
    "property": 8,
    "scene": 2
}

IMAGE_ROOT = "data/raw/physbench_full/image"

# safety filter (prevents 12MP crash like idx=9256)
MAX_MEGA_PIXELS = 3.0


# -----------------------------
# ICOT BIAS SCORE (kept)
# -----------------------------
def icot_score(item):
    q = (item.get("question") or "").lower()
    t = item.get("task_type", "")

    score = 0

    if any(k in q for k in ["push", "pull", "collision", "force", "move", "hit"]):
        score += 3

    if any(k in q for k in ["sequence", "order", "before", "after"]):
        score += 3

    if any(k in q for k in ["behind", "hidden", "partially", "obscured"]):
        score += 3

    if any(k in q for k in ["balance", "tilt", "fall", "stable", "support"]):
        score += 2

    if t == "relationships":
        score += 1

    if "how many" in q or "count" in q:
        score -= 3

    return score


# -----------------------------
# safe image filtering only
# -----------------------------
def load_and_filter():
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    cleaned = []
    removed = 0

    for item in data:
        if item.get("answer") not in VALID_ANSWERS:
            removed += 1
            continue

        if not item.get("image_files"):
            removed += 1
            continue

        ok = True

        for f in item["image_files"]:
            path = os.path.join(IMAGE_ROOT, f)

            if not os.path.exists(path):
                ok = False
                break

            try:
                img = Image.open(path)
                w, h = img.size
                mp = (w * h) / 1e6

                if mp > MAX_MEGA_PIXELS:
                    ok = False
                    break

            except Exception:
                ok = False
                break

        if not ok:
            removed += 1
            continue

        cleaned.append(item)

    print(f"Removed: {removed}")
    print(f"Remaining: {len(cleaned)}")

    return cleaned


# -----------------------------
# grouping
# -----------------------------
def group_by_task(data):
    groups = defaultdict(list)
    for item in data:
        groups[item["task_type"]].append(item)
    return groups


# -----------------------------
# ICOT-biased sampling (NO DUPLICATES FIXED)
# -----------------------------
def biased_sample(pool, n):
    scored = [(x, icot_score(x)) for x in pool]
    scored.sort(key=lambda x: x[1], reverse=True)

    top_pool = [x[0] for x in scored[: max(n * 3, len(scored) // 2)]]

    # FIX: sample WITHOUT replacement
    return random.sample(top_pool, k=n)


# -----------------------------
# stratified sampling
# -----------------------------
def sample_stratified(groups):
    sampled = []

    for task_type, n in TARGET_DISTRIBUTION.items():
        pool = groups.get(task_type, [])

        if len(pool) < n:
            raise ValueError(f"Not enough samples for {task_type}")

        sampled.extend(biased_sample(pool, n))

    return sampled


# -----------------------------
# main
# -----------------------------
def main():
    print("Loading dataset...")
    data = load_and_filter()

    print("Grouping...")
    groups = group_by_task(data)

    print("Available distribution:")
    for k, v in groups.items():
        print(k, len(v))

    print("Sampling ICOT-biased 50-sample set (50% dynamics)...")
    subset = sample_stratified(groups)

    print("Final subset:", len(subset))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(subset, f, indent=2)

    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
