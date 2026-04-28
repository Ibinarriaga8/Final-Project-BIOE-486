import kagglehub
import pandas as pd
import os
from glob import glob
import shutil

SOURCE_PATH = "/home/jorge/.cache/kagglehub/datasets/jasonroggy/grazpedwri-dx/versions/1"
TARGET_PATH = "data/GRAZ"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")



# Download latest version
def download_dataset():
    path = kagglehub.dataset_download("jasonroggy/grazpedwri-dx")

    print("Path to dataset files:", path)

def load_data(path):
    

    data_file = f"{path}/data.csv"  # Adjust the file name as needed
    data = pd.read_csv(data_file)

    return data


def load_images(source_path, target_path):
    os.makedirs(target_path, exist_ok=True)
    os.makedirs(os.path.join(target_path, "images"), exist_ok=True)

    csv_src = os.path.join(source_path, "dataset.csv")
    csv_dst = os.path.join(target_path, "dataset.csv")
    shutil.copy2(csv_src, csv_dst)

    image_dirs = [
        os.path.join(source_path, d)
        for d in os.listdir(source_path)
        if d.startswith("images_part") and os.path.isdir(os.path.join(source_path, d))
    ]

    image_paths = []
    for image_dir in image_dirs:
        image_paths.extend(glob(os.path.join(image_dir, "**", "*.*"), recursive=True))

    image_paths = [
        p for p in image_paths
        if p.lower().endswith(IMAGE_EXTENSIONS)
    ]

    for src in image_paths:
        dst = os.path.join(target_path, "images", os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    print(f"Copied dataset.csv to {csv_dst}")
    print(f"Copied {len(image_paths)} images to {os.path.join(target_path, 'images')}")


if __name__ == "__main__":
    load_images(SOURCE_PATH, TARGET_PATH)
