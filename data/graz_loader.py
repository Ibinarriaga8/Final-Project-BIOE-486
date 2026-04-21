import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image


class GRAZDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["full_path"])
        image = np.array(image).astype(np.float32)


        image -= image.min()
        max_value = image.max()
        
        if max_value > 0:
            image /= max_value

        image = Image.fromarray((image * 255).astype(np.uint8))

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label"])
        return image, label


def load_graz_metadata(graz_root="data/GRAZ"):
    csv_path = os.path.join(graz_root, "dataset.csv")
    df = pd.read_csv(csv_path)

    image_dir = os.path.join(graz_root, "images")
    image_files = [
        filename for filename in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, filename))
    ]

    stem_to_path = {
        Path(filename).stem: os.path.join(image_dir, filename)
        for filename in image_files
    }

    df["full_path"] = df["filestem"].map(stem_to_path)
    df = df[df["full_path"].notna()].copy()

    df["label"] = df["fracture_visible"].fillna(0).astype(int)

    return df


def get_graz_loader(batch_size=16, img_size=256, num_workers=0):
    df = load_graz_metadata()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    dataset = GRAZDataset(df, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return loader


def test_graz_loader():
    loader = get_graz_loader(batch_size=4, img_size=128, num_workers=0)
    images, labels = next(iter(loader))

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)


def save_sample_images():
    loader = get_graz_loader(batch_size=4, img_size=128, num_workers=0)
    images, labels = next(iter(loader))

    os.makedirs("debug_graz", exist_ok=True)

    for i in range(len(images)):
        path = f"debug_graz/sample_{i}_label_{labels[i].item()}.png"
        save_image(images[i], path, normalize=True)

    print("Saved images in ./debug_graz/")


if __name__ == "__main__":
    test_graz_loader()
    save_sample_images()