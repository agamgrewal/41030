import os
import random
import shutil

def trim_dataset(source_folder, dest_folder, num_images=5000):

    image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} images in {source_folder}")

    subset = random.sample(image_files, min(num_images, len(image_files)))

    os.makedirs(dest_folder, exist_ok=True)

    for img in subset:
        shutil.copy(os.path.join(source_folder, img), os.path.join(dest_folder, img))

    print(f"Sample dataset created: {len(subset)} images saved in {dest_folder}")

trim_dataset(
    source_folder="data/train2017",   # Folder containing your large dataset
    dest_folder="data/sample5000",    # Folder where 5000 random images will go
    num_images=5000                   # Number of images to keep
)