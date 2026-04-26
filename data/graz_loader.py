import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PIN_MEMORY = torch.cuda.is_available()


class GRAZDataset(Dataset):
    def __init__(self, df, transform=None, return_label=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.return_label = return_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["full_path"]).convert("L")
        image = np.array(image, dtype=np.float32)

        image -= image.min()
        max_value = image.max()
        if max_value > 0:
            image /= max_value

        image = Image.fromarray((image * 255).astype(np.uint8))

        if self.transform is not None:
            image = self.transform(image)

        if self.return_label:
            return image, int(row["label"])
        return image


def load_graz_metadata(graz_root="data/GRAZ"):
    csv_path = os.path.join(graz_root, "dataset.csv")
    df = pd.read_csv(csv_path)

    part_dirs = sorted(
        os.path.join(graz_root, d)
        for d in os.listdir(graz_root)
        if d.startswith("images_part") and os.path.isdir(os.path.join(graz_root, d))
    )

    stem_to_path = {}
    for image_dir in part_dirs:
        for filename in os.listdir(image_dir):
            full_path = os.path.join(image_dir, filename)
            if os.path.isfile(full_path):
                stem_to_path[Path(filename).stem] = full_path

    df["full_path"] = df["filestem"].map(stem_to_path)
    df = df[df["full_path"].notna()].copy()
    df["label"] = df["fracture_visible"].fillna(0).astype(int)

    return df


def get_graz_loaders(batch_size=16, img_size=256, num_workers=0, val_split=0.2, seed=42):
    df = load_graz_metadata()

    train_df, val_df = train_test_split(
        df, test_size=val_split, random_state=seed, stratify=df["label"]
    )

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dataset = GRAZDataset(train_df, transform=train_transform)
    val_dataset = GRAZDataset(val_df, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=PIN_MEMORY)

    return train_loader, val_loader


def get_graz_recon_datasets(img_size=256, val_split=0.2, seed=42):
    """Returns (train_dataset, val_dataset) with only normal (label=0) images, no labels."""
    df = load_graz_metadata()
    normal_df = df[df["label"] == 0].copy()

    train_df, val_df = train_test_split(normal_df, test_size=val_split, random_state=seed)

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    return (
        GRAZDataset(train_df, transform=train_transform, return_label=False),
        GRAZDataset(val_df, transform=val_transform, return_label=False),
    )


def get_graz_loader(batch_size=16, img_size=256, num_workers=0):
    """Single loader with all data (kept for backwards compatibility)."""
    df = load_graz_metadata()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    dataset = GRAZDataset(df, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


if __name__ == "__main__":
    train_loader, val_loader = get_graz_loaders(batch_size=4, img_size=128)

    images, labels = next(iter(train_loader))
    print("Train — images:", images.shape, "labels:", labels)

    images, labels = next(iter(val_loader))
    print("Val   — images:", images.shape, "labels:", labels)

    train_ds, val_ds = get_graz_recon_datasets(img_size=128)
    print(f"Recon train: {len(train_ds)} normal samples")
    print(f"Recon val:   {len(val_ds)} normal samples")
