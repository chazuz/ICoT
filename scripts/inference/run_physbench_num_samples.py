import json
import os
import time
import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DATA_PATH = "data/processed/physbench_50_seed0.json"
IMAGE_ROOT = "data/raw/physbench_full/image"


# -----------------------------
# logging
# -----------------------------
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -----------------------------
# dataset
# -----------------------------
def load_data():
    with open(DATA_PATH, "r") as f:
        return json.load(f)


# -----------------------------
# image loader + diagnostics
# -----------------------------
def load_images(file_list):
    imgs = []
    meta = []

    for f in file_list:
        path = os.path.join(IMAGE_ROOT, f)

        if not os.path.exists(path):
            log(f"[WARN] missing image: {path}")
            continue

        t0 = time.time()
        img = Image.open(path).convert("RGB")
        dt = time.time() - t0

        w, h = img.size
        mp = (w * h) / 1e6

        log(f"  [IMG] {f} | {w}x{h} | {mp:.2f} MP | load {dt:.2f}s")

        imgs.append(img)
        meta.append((w, h, mp))

    return imgs, meta


# -----------------------------
# prompt
# -----------------------------
def build_prompt(question):
    return "Answer A, B, C, or D.\n\n" + question


# -----------------------------
# main
# -----------------------------
def main(args):

    log("Loading model + processor...")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

    model.eval()

    log("Model loaded")

    data = load_data()

    total = len(data)
    n = args.num_samples if args.num_samples is not None else total

    log(f"Loaded {total} samples")
    log(f"Running first {n} samples")

    # -----------------------------
    # loop
    # -----------------------------
    for i, sample in enumerate(data[:n]):

        log("\n" + "=" * 60)
        log(f"SAMPLE {i} | IDX {sample['idx']}")
        log("=" * 60)

        # -------- image load --------
        t_img = time.time()
        images, meta = load_images(sample["image_files"])
        t_img = time.time() - t_img

        log(f"Loaded {len(images)} images in {t_img:.2f}s")

        if len(images) == 0:
            log("[SKIP] no images")
            continue

        total_mp = sum(m[2] for m in meta)
        log(f"TOTAL VISUAL LOAD: {total_mp:.2f} MP")

        prompt = build_prompt(sample["question"])

        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in images],
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # -------- tokenization --------
        log("Tokenizing...")

        t_tok = time.time()

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        t_tok = time.time() - t_tok

        log(f"Tokenization done in {t_tok:.2f}s")

        # DEBUG: tensor shapes
        for k, v in inputs.items():
            try:
                log(f"  INPUT {k}: shape={tuple(v.shape)} dtype={v.dtype}")
            except Exception:
                pass

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # -------- generation --------
        log("Running inference...")

        t_gen = time.time()

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=10
            )

        t_gen = time.time() - t_gen

        log(f"Generation done in {t_gen:.2f}s")

        # decode
        result = processor.decode(output[0], skip_special_tokens=True)

        print("\nQUESTION:", sample["question"])
        print("GROUND TRUTH:", sample.get("answer"))
        print("MODEL OUTPUT:", result)
        print("-" * 60)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=None)

    args = parser.parse_args()
    main(args)
