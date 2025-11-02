# baselinerun.py
# Agam Grewal – Capstone Baseline VQA (5000 images)

import os
import json
import re
from tqdm import tqdm
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering

# =====================================================
# 🔹 PATH CONFIGURATION
# =====================================================
IMAGE_DIR = "src/data/sample5000"
QUESTION_PATH = "src/data/annotations/vqa_train_sample_questions.json"
ANNOTATION_PATH = "src/data/annotations/vqa_train_sample5000_annotations.json"
OUTPUT_PATH = "results/baseline_results_5000.json"

# =====================================================
# 🔹 LOAD MODEL AND PROCESSOR
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Loading BLIP VQA model on {device}...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
model.eval()

# =====================================================
# 🔹 LOAD DATA
# =====================================================
print(f"🔹 Loading questions from {QUESTION_PATH}")
with open(QUESTION_PATH) as f:
    q_data = json.load(f)["questions"]

print(f"🔹 Loading annotations (answers) from {ANNOTATION_PATH}")
with open(ANNOTATION_PATH) as f:
    a_data = json.load(f)["annotations"]

# Map answers by question_id
answer_map = {a["question_id"]: a["multiple_choice_answer"] for a in a_data}

# Merge question text with ground truth
merged = []
for q in q_data:
    qid = q["question_id"]
    if qid in answer_map:
        merged.append({
            "image_id": q["image_id"],
            "question_id": qid,
            "question": q["question"],
            "answer": answer_map[qid]
        })

print(f"✅ Merged {len(merged)} question–answer pairs.")

# =====================================================
# 🔹 RUN BASELINE INFERENCE
# =====================================================
results = []
missing = 0

for item in tqdm(merged, desc="Running BLIP VQA baseline"):
    img_id = item["image_id"]
    qid = item["question_id"]
    qtext = item["question"]

    img_path = os.path.join(IMAGE_DIR, f"{img_id:012d}.jpg")
    if not os.path.exists(img_path):
        missing += 1
        continue

    try:
        raw_image = Image.open(img_path).convert("RGB")
        inputs = processor(raw_image, qtext, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=10)
        pred = processor.decode(output[0], skip_special_tokens=True)
        results.append({"question_id": qid, "answer": pred})
    except Exception as e:
        print(f"⚠️ Error on {img_id}: {e}")

# Save predictions
os.makedirs("results", exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Saved {len(results)} predictions to {OUTPUT_PATH}")
print(f"⚠️ Missing images: {missing}")

# =====================================================
# 🔹 EVALUATE AGAINST GROUND TRUTH
# =====================================================
def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s

gt_map = {m["question_id"]: normalize(m["answer"]) for m in merged}

correct = 0
total = 0
missing_pred = 0

for p in results:
    qid = p["question_id"]
    if qid not in gt_map:
        missing_pred += 1
        continue
    total += 1
    if normalize(p["answer"]) == gt_map[qid]:
        correct += 1

accuracy = (correct / total * 100) if total > 0 else 0

print("\n=====================================")
print(f"✅ Evaluated {total} questions")
print(f"✅ Correct answers: {correct}")
print(f"⚠️ Missing predictions: {missing_pred}")
print(f"🎯 Accuracy: {accuracy:.2f}%")
print("=====================================")

# Save summary
summary = {
    "evaluated": total,
    "correct": correct,
    "missing_predictions": missing_pred,
    "accuracy": round(accuracy, 2),
}
with open("results/baseline_accuracy_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("📁 Saved evaluation summary to results/baseline_accuracy_summary.json")
