import json
import os
import time
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

DATA_PATH = "data/processed/physbench_50_seed0.json"
IMAGE_ROOT = "data/raw/physbench_full/image"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_sample():
    with open(DATA_PATH, "r") as f:
        return json.load(f)[0]


def load_images(files):
    imgs = []
    for f in files:
        p = os.path.join(IMAGE_ROOT, f)
        if os.path.exists(p):
            imgs.append(Image.open(p).convert("RGB"))
        else:
            log(f"[WARN] missing {p}")
    return imgs


def main():

    log("Loading processor + model...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

    log("Model loaded")

    sample = load_sample()

    log(f"IDX: {sample['idx']}")
    log(f"Keys: {list(sample.keys())}")

    images = load_images(sample["image_files"])
    log(f"Loaded images: {len(images)}")

    prompt = (
        "Answer A, B, C, or D.\n\n"
        f"{sample['question']}"
    )

    # ---------------- FIX HERE ----------------
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in images],
                {"type": "text", "text": prompt},
            ],
        }
    ]

    log("Tokenizing chat...")

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,              # 🔥 CRITICAL FIX
        return_dict=True,           # 🔥 CRITICAL FIX
        return_tensors="pt"
    )

    log(f"Input keys: {inputs.keys()}")

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    log("Running inference...")

    start = time.time()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=10
        )

    log(f"Done in {time.time() - start:.2f}s")

    result = processor.decode(output[0], skip_special_tokens=True)

    print("\n====================")
    print("QUESTION:")
    print(sample["question"])
    print("\nGROUND TRUTH:")
    print(sample["answer"])
    print("\nMODEL OUTPUT:")
    print(result)
    print("====================")


if __name__ == "__main__":
    main()
