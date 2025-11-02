import os
import json
from tqdm import tqdm
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering

# ====================================================
# CONFIGURATION
# ====================================================
QUESTIONS_FILE = "/data/akgrewal/pycharm_projects/capstone/src/data/annotations/vqa_train_sample_questions.json"
IMAGE_DIR = "/data/akgrewal/pycharm_projects/capstone/src/data/sample5000"
RESULTS_FILE = "/data/akgrewal/pycharm_projects/capstone/results/baseline_results.json"

# ====================================================
# LOAD MODEL + PROCESSOR
# ====================================================
print("🔹 Loading BLIP VQA model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
model.eval()

# ====================================================
# LOAD QUESTIONS FILE
# ====================================================
print(f"🔹 Loading questions from {QUESTIONS_FILE}")
with open(QUESTIONS_FILE, "r") as f:
    raw = json.load(f)

if isinstance(raw, dict) and "questions" in raw:
    data = raw["questions"]
elif isinstance(raw, list):
    data = raw
else:
    raise ValueError("❌ Invalid JSON structure: expected a list or 'questions' key.")

print(f"✅ Loaded {len(data)} questions.")

# ====================================================
# RUN INFERENCE
# ====================================================
predictions = []
missing_images = 0

for item in tqdm(data):
    img_id = item.get("image_id")
    question = item.get("question")
    qid = item.get("question_id")

    # Try both plain and zero-padded image names
    image_path = os.path.join(IMAGE_DIR, f"{img_id}.jpg")
    if not os.path.exists(image_path):
        image_path = os.path.join(IMAGE_DIR, f"{img_id:012d}.jpg")
    if not os.path.exists(image_path):
        missing_images += 1
        continue

    try:
        raw_image = Image.open(image_path).convert("RGB")
    except Exception:
        missing_images += 1
        continue

    # Prepare inputs
    inputs = processor(raw_image, question, return_tensors="pt").to(device)

    # ✅ Correct inference call using generate()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    predictions.append({
        "question_id": qid,
        "answer": answer
    })

# ====================================================
# SAVE RESULTS
# ====================================================
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
with open(RESULTS_FILE, "w") as f:
    json.dump(predictions, f, indent=2)

print(f"\n✅ Saved {len(predictions)} predictions to {RESULTS_FILE}")
print(f"⚠️ Missing images: {missing_images}")
