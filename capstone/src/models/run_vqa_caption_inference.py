# run_vqa_caption_inference.py
# Agam Grewal – Capstone: Visual Question Answering with Caption Augmentation

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
CAPTION_FILE = "src/data/annotations/captions_sample5000.jsonl"
QUESTION_PATH = "src/data/annotations/vqa_train_sample5000_questions.json"
ANNOTATION_PATH = "src/data/annotations/vqa_train_sample5000_annotations.json"
OUTPUT_PATH = "results/caption_predictions.json"
SUMMARY_PATH = "results/caption_accuracy_summary.json"

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

# Load captions
captions = {}
with open(CAPTION_FILE) as f:
    for line in f:
        entry = json.loads(line)
        image_id = int(os.path.splitext(entry["image_id"])[0])
        captions[image_id] = entry["caption"]

answer_map = {a["question_id"]: a["multiple_choice_answer"] for a in a_data}
merged = [
    {
        "image_id": q["image_id"],
        "question_id": q["question_id"],
        "question": q["question"],
        "caption": captions.get(q["image_id"], ""),
        "answer": answer_map[q["question_id"]],
    }
    for q in q_data if q["question_id"] in answer_map
]

print(f"Loaded {len(merged)} question–answer–caption triples.")

# =====================================================
# RUN INFERENCE WITH CAPTIONS
# =====================================================
results = []
missing = 0

for item in tqdm(merged, desc="Running BLIP-VQA with captions"):
    img_path = os.path.join(IMAGE_DIR, f"{item['image_id']:012d}.jpg")
    if not os.path.exists(img_path):
        missing += 1
        continue
    try:
        image = Image.open(img_path).convert("RGB")
        combined_text = f"Caption: {item['caption']} Question: {item['question']}"
        inputs = processor(image, combined_text, return_tensors="pt").to(device)
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

print(f"Caption-Augmented Accuracy: {accuracy:.2f}%")
