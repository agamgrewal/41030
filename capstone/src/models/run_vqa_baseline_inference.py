# run_vqa_baseline_inference.py
# Agam Grewal – Capstone: Baseline Visual Question Answering (No Captions)

import os
import json
import re
from tqdm import tqdm
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering

# =====================================================
# CONFIGURATION
# =====================================================
IMAGE_DIR = "src/data/sample5000"
QUESTION_PATH = "src/data/annotations/vqa_train_sample5000_questions.json"
ANNOTATION_PATH = "src/data/annotations/vqa_train_sample5000_annotations.json"
OUTPUT_PATH = "results/baseline_predictions.json"
SUMMARY_PATH = "results/baseline_accuracy_summary.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading BLIP VQA model on {device}...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
model.eval()

# =====================================================
# LOAD DATA
# =====================================================
with open(QUESTION_PATH) as f:
    q_data = json.load(f)["questions"]

with open(ANNOTATION_PATH) as f:
    a_data = json.load(f)["annotations"]

answer_map = {a["question_id"]: a["multiple_choice_answer"] for a in a_data}
merged = [
    {
        "image_id": q["image_id"],
        "question_id": q["question_id"],
        "question": q["question"],
        "answer": answer_map[q["question_id"]],
    }
    for q in q_data if q["question_id"] in answer_map
]

print(f"Loaded {len(merged)} question–answer pairs.")

# =====================================================
# RUN BASELINE INFERENCE
# =====================================================
results = []
missing = 0

for item in tqdm(merged, desc="Running baseline BLIP-VQA"):
    img_path = os.path.join(IMAGE_DIR, f"{item['image_id']:012d}.jpg")
    if not os.path.exists(img_path):
        missing += 1
        continue
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(image, item["question"], return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=10)
        pred = processor.decode(output[0], skip_special_tokens=True)
        results.append({"question_id": item["question_id"], "answer": pred})
    except Exception as e:
        print(f"Error on {item['image_id']}: {e}")

os.makedirs("results", exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved {len(results)} predictions to {OUTPUT_PATH}")
print(f"Missing images: {missing}")

# =====================================================
# EVALUATE ACCURACY
# =====================================================
def normalize(s):
    return re.sub(r"[^a-z0-9 ]+", "", s.lower().strip())

gt = {m["question_id"]: normalize(m["answer"]) for m in merged}
correct = sum(normalize(p["answer"]) == gt.get(p["question_id"], "") for p in results)
accuracy = correct / len(results) * 100

summary = {
    "evaluated": len(results),
    "correct": correct,
    "accuracy": round(accuracy, 2),
}
with open(SUMMARY_PATH, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Baseline Accuracy: {accuracy:.2f}%")