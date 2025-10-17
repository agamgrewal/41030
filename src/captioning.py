import os
import json
import random
from tqdm import tqdm
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

def generate_and_store_captions(input_folder, output_jsonl, limit=10):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    model.eval()

    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    random.shuffle(image_files)

    image_files = image_files[:limit]
    print(f"Found {len(image_files)} images (limited to {limit}) in {input_folder}")

    with open(output_jsonl, "w") as f:
        for img_name in tqdm(image_files, desc=f"Generating captions for {os.path.basename(input_folder)}"):
            try:
                image_path = os.path.join(input_folder, img_name)
                image = Image.open(image_path).convert("RGB")

                inputs = processor(images=image, return_tensors="pt").to(device)
                out = model.generate(**inputs, max_new_tokens=30)
                caption = processor.decode(out[0], skip_special_tokens=True)

                record = {"image_id": img_name, "caption": caption}
                f.write(json.dumps(record) + "\n")
            except Exception as e:
                print(f"⚠️ Skipping {img_name}: {e}")

    print(f"\n✅ Captions saved to {output_jsonl}")

if __name__ == "__main__":
    generate_and_store_captions(
        input_folder="src/data/train2017",
        output_jsonl="src/data/",
        limit=10
    )
