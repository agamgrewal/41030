# qualitative_examples.py
# Agam Grewal – Capstone Project
# Selects qualitative examples where captions caused incorrect answers (Figure 4.5)

import os
import json
import random
import matplotlib.pyplot as plt
from PIL import Image

# ------------------------------------------------------
# Paths
# ------------------------------------------------------
IMAGE_DIR = "src/data/sample5000"
QUESTION_FILE = "src/data/annotations/vqa_train_sample5000_questions.json"
ANNOTATION_FILE = "src/data/annotations/vqa_train_sample5000_annotations.json"
BASELINE_RESULTS = "results/baseline_predictions.json"
CAPTION_RESULTS = "results/caption_predictions.json"
CAPTION_FILE = "src/data/annotations/captions_sample5000.jsonl"

OUTPUT_FIGURE = "results/figure4_5_caption_failures.png"

# ------------------------------------------------------
# Load data
# ------------------------------------------------------
with open(QUESTION_FILE) as f:
    questions = json.load(f)["questions"]

with open(ANNOTATION_FILE) as f:
    annotations = json.load(f)["annotations"]

with open(BASELINE_RESULTS) as f:
    baseline_preds = {p["question_id"]: p["answer"].strip().lower() for p in json.load(f)}

with open(CAPTION_RESULTS) as f:
    caption_preds = {p["question_id"]: p["answer"].strip().lower() for p in json.load(f)}

# Load captions
captions = {}
with open(CAPTION_FILE) as f:
    for line in f:
        entry = json.loads(line)
        img_id = int(os.path.splitext(entry["image_id"])[0])
        captions[img_id] = entry["caption"]

# Ground truth answers
answers = {a["question_id"]: a["multiple_choice_answer"].strip().lower() for a in annotations}

# ------------------------------------------------------
# Find cases where baseline is correct but caption is wrong
# ------------------------------------------------------
failures = []
for qid, gt in answers.items():
    base = baseline_preds.get(qid)
    cap = caption_preds.get(qid)
    if not base or not cap:
        continue
    if base == gt and cap != gt:
        failures.append(qid)

if not failures:
    print("⚠️ No examples found where caption failed while baseline was correct.")
    exit()

# Randomly sample up to 4 of these
sample_qids = random.sample(failures, min(4, len(failures)))

# ------------------------------------------------------
# Plot grid
# ------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, qid in enumerate(sample_qids):
    q = next(q for q in questions if q["question_id"] == qid)
    img_id = q["image_id"]
    img_path = os.path.join(IMAGE_DIR, f"{img_id:012d}.jpg")

    if not os.path.exists(img_path):
        continue

    image = Image.open(img_path).convert("RGB")
    axes[i].imshow(image)
    axes[i].axis("off")

    gt = answers[qid]
    base = baseline_preds[qid]
    cap = caption_preds[qid]
    caption_text = captions.get(img_id, "")

    title = (
        f"Q: {q['question']}\n"
        f"GT: {gt}\n"
        f"Baseline ✓: {base}\n"
        f"Caption ✗: {cap}\n"
        f"Image Caption: {caption_text}"
    )

    axes[i].set_title(title, fontsize=9, loc="left", pad=10)

plt.suptitle("Qualitative Examples: Caption Failures vs Baseline Correct", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_FIGURE, dpi=300)
plt.close()

print(f" Saved figure showing caption model failures to {OUTPUT_FIGURE}")
