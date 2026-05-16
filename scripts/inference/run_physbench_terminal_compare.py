import json
import argparse
import torch
from PIL import Image
import os
import re
import random
import time
import signal
from datetime import datetime
from collections import defaultdict

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_NAME  = "Qwen/Qwen2.5-VL-3B-Instruct"
BASE_PATH   = "data/processed/physbench_50_seed0.json"
ICOT_PATH   = "data/processed/physbench_50_seed0_icot.json"
IMAGE_ROOT  = "data/raw/physbench_full/image"

# Large images are the #1 cause of hanging on CPU
# Cap each image at this many pixels (width * height)
MAX_PIXELS  = 200_000
TIMEOUT_SEC = 120   # skip sample if inference takes longer than this


# -------------------------
# timeout via SIGALRM (Linux only)
# -------------------------
class TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutError()


# -------------------------
# data
# -------------------------
def load_json(path):
    with open(path) as f:
        return json.load(f)


# -------------------------
# images — resize to MAX_PIXELS
# -------------------------
def resize_image(img, max_pixels=MAX_PIXELS):
    w, h = img.size
    total = w * h
    if total > max_pixels:
        scale = (max_pixels / total) ** 0.5
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img

def load_images(file_list):
    imgs = []
    for f in file_list:
        p = os.path.join(IMAGE_ROOT, f)
        if os.path.exists(p):
            img = Image.open(p).convert("RGB")
            img = resize_image(img)
            imgs.append(img)
        else:
            print(f"  [WARN] missing: {p}")
    return imgs


# -------------------------
# prompt
# -------------------------
def prompt(q):
    return f"Answer with exactly one letter: A, B, C, or D.\n\n{q}"


# -------------------------
# answer extraction
# -------------------------
def extract_answer(text):
    matches = re.findall(r"\b([AaBbCcDd])\b", text)
    return matches[-1].upper() if matches else "?"


# -------------------------
# inference
# -------------------------
def run(model, processor, images, text):
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": img} for img in images]
                 + [{"type": "text",  "text":  text}]
    }]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    return extract_answer(processor.decode(out[0], skip_special_tokens=True))


# -------------------------
# main
# -------------------------
def main(args):
    random.seed(args.seed)

    experiment_start = time.time()
    print(f"Experiment start : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seed             : {args.seed}")
    print(f"Max pixels/image : {MAX_PIXELS:,}")
    print(f"Timeout/sample   : {TIMEOUT_SEC}s")
    print()

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map="cpu"
    ).eval()
    print("Model ready.\n")

    base_data = load_json(BASE_PATH)
    icot_data = load_json(ICOT_PATH)

    icot_map = {item["idx"]: item for item in icot_data}
    pairs    = [(b, icot_map[b["idx"]]) for b in base_data if b["idx"] in icot_map]

    n = min(args.num_samples, len(pairs))

    # always randomize order
    sampled = random.sample(pairs, n)

    results      = []
    sample_times = []
    skipped      = []

    base_correct = icot_correct = 0
    both_correct = neither_correct = base_only = icot_only = 0

    for i, (b, c) in enumerate(sampled):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [{i+1}/{n}] idx {b['idx']:>5} | GT: {b['answer']} | running...", flush=True)

        t_start = time.time()

        try:
            # arm timeout
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(TIMEOUT_SEC)

            images    = load_images(b["image_files"])
            gt        = b["answer"]
            base_pred = run(model, processor, images, prompt(b["question"]))
            icot_pred = run(model, processor, images, prompt(c["question"]))

            signal.alarm(0)  # disarm

        except TimeoutError:
            signal.alarm(0)
            t_elapsed = time.time() - t_start
            print(f"[{ts}] [{i+1}/{n}] idx {b['idx']:>5} | SKIPPED (timeout {t_elapsed:.0f}s)", flush=True)
            skipped.append(b["idx"])
            continue

        except Exception as e:
            signal.alarm(0)
            print(f"[{ts}] [{i+1}/{n}] idx {b['idx']:>5} | SKIPPED (error: {e})", flush=True)
            skipped.append(b["idx"])
            continue

        t_elapsed = time.time() - t_start
        sample_times.append(t_elapsed)

        b_ok = base_pred == gt
        i_ok = icot_pred == gt

        base_correct    += b_ok
        icot_correct    += i_ok
        both_correct    += b_ok and i_ok
        neither_correct += not b_ok and not i_ok
        base_only       += b_ok and not i_ok
        icot_only       += i_ok and not b_ok

        outcome = "BOTH" if (b_ok and i_ok) else \
                  "BASE" if (b_ok and not i_ok) else \
                  "ICOT" if (i_ok and not b_ok) else "NONE"

        results.append({
            "idx":       b["idx"],
            "gt":        gt,
            "base_pred": base_pred,
            "icot_pred": icot_pred,
            "b_ok":      b_ok,
            "i_ok":      i_ok,
            "outcome":   outcome,
            "task_type": b.get("task_type", "?"),
            "sub_type":  b.get("sub_type",  "?"),
        })

        marker = " <- BASE wins" if outcome == "BASE" else \
                 " <- ICOT wins" if outcome == "ICOT" else ""

        print(f"[{ts}] [{i+1}/{n}] idx {b['idx']:>5} | GT: {gt} | BASE: {base_pred} | ICOT: {icot_pred} | {t_elapsed:.1f}s{marker}", flush=True)

    # =========================================================
    # FULL EVALUATION
    # =========================================================
    total_time = time.time() - experiment_start
    evaluated  = len(results)

    print()
    print("=" * 60)
    print("  EXPERIMENT SUMMARY")
    print(f"  Finished  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration  : {total_time:.1f}s total", end="")
    if sample_times:
        print(f"  |  avg {sum(sample_times)/len(sample_times):.1f}s/sample")
    else:
        print()
    print(f"  Evaluated : {evaluated}/{n}")
    if skipped:
        print(f"  Skipped   : {len(skipped)}  {skipped}")
    print("=" * 60)

    if evaluated == 0:
        print("  No samples evaluated.")
        return

    # --- Accuracy ---
    print()
    print("  ACCURACY")
    print(f"  BASE : {base_correct}/{evaluated}  ({100*base_correct/evaluated:.1f}%)")
    print(f"  ICOT : {icot_correct}/{evaluated}  ({100*icot_correct/evaluated:.1f}%)")
    delta     = icot_correct - base_correct
    delta_pct = 100 * delta / evaluated
    direction = "improvement" if delta > 0 else "regression" if delta < 0 else "no change"
    print(f"  Delta: {delta:+d} ({delta_pct:+.1f}%)  ->  {direction}")

    # --- Agreement ---
    print()
    print("  AGREEMENT")
    print(f"  Both correct      : {both_correct:>3}  ({100*both_correct/evaluated:.1f}%)")
    print(f"  Neither correct   : {neither_correct:>3}  ({100*neither_correct/evaluated:.1f}%)")
    print(f"  BASE only correct : {base_only:>3}  ({100*base_only/evaluated:.1f}%)")
    print(f"  ICOT only correct : {icot_only:>3}  ({100*icot_only/evaluated:.1f}%)")

    # --- Relative performance ---
    contested = base_only + icot_only
    print()
    print("  RELATIVE PERFORMANCE  (samples where strategies diverge)")
    if contested > 0:
        print(f"  Contested samples : {contested}")
        print(f"  BASE wins         : {base_only}  ({100*base_only/contested:.1f}% of contested)")
        print(f"  ICOT wins         : {icot_only}  ({100*icot_only/contested:.1f}% of contested)")
    else:
        print("  No contested samples — strategies always agree.")

    same_answer = sum(1 for r in results if r["base_pred"] == r["icot_pred"])
    print(f"  Same prediction   : {same_answer}/{evaluated}  ({100*same_answer/evaluated:.1f}%)")

    # --- Per task_type breakdown ---
    task_stats = defaultdict(lambda: {"n": 0, "base": 0, "icot": 0})
    for r in results:
        tt = r["task_type"]
        task_stats[tt]["n"]    += 1
        task_stats[tt]["base"] += r["b_ok"]
        task_stats[tt]["icot"] += r["i_ok"]

    if len(task_stats) >= 1:
        print()
        print("  BREAKDOWN BY TASK TYPE")
        print(f"  {'task_type':<18} {'n':>3}  {'BASE':>6}  {'ICOT':>6}  {'delta':>6}")
        print("  " + "-" * 46)
        for tt, s in sorted(task_stats.items()):
            nn    = s["n"]
            b_acc = 100 * s["base"] / nn
            i_acc = 100 * s["icot"] / nn
            d     = s["icot"] - s["base"]
            print(f"  {tt:<18} {nn:>3}  {b_acc:>5.1f}%  {i_acc:>5.1f}%  {d:>+5d}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()
    main(args)
