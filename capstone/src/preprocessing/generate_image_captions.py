# generate_image_captions.py
# Author: Agam Grewal
# Project: Image Captioning as a Data Augmentation Strategy for VQA
# Purpose: Generate BLIP captions for the 5000-image dataset

import os
import json
from tqdm import tqdm
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# =====================================================
# CONFIGURATION
# =====================================================
IMAGE_DIR = "src/data/sample5000"
OUTPUT_FILE = "src/data/annotations/captions_sample5000.jsonl"

# =====================================================
# LOAD MODEL
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading BLIP captioning model on {device}...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
model.eval()

# =====================================================
# GENERATE CAPTIONS
# =====================================================
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w") as out_f:
    for fname in tqdm(sorted(os.listdir(IMAGE_DIR))):
        if not fname.endswith(".jpg"):
            continue

        img_path = os.path.join(IMAGE_DIR, fname)
        try:
            image = Image.open(img_path).convert("RGB")

            inputs = processor(image, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=20)
            caption = processor.decode(output[0], skip_special_tokens=True)

            out_f.write(json.dumps({"image_id": fname, "caption": caption}) + "\n")

        except Exception as e:
            print(f"Error processing {fname}: {e}")

print(f"\n Captions saved to {OUTPUT_FILE}")
