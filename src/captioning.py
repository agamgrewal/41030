import os
import json
from tqdm import tqdm
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

def generate_and_store_captions(input_folder, output_jsonl):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    model.eval()

    image_files = sorted([f for f in os.listdir(input_folder)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"Found {len(image_files)} images in {input_folder}")

    results = []
    for img_name in tqdm(image_files, desc="Generating captions"):
        img_path = os.path.join(input_folder, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
            inputs = processor(image, return_tensors="pt").to(device)

            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=30)

            caption = processor.decode(output[0], skip_special_tokens=True)
            results.append({"image_id": img_name, "caption": caption})

        except Exception as e:
            print(f"Skipped {img_name}: {e}")

    with open(output_jsonl, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    print(f"\n Saved {len(results)} captions to {output_jsonl}")

if __name__ == "__main__":
    generate_and_store_captions(
        input_folder="data/sample5000",
        output_jsonl="data/blip_sample5000_captions.jsonl"
    )
