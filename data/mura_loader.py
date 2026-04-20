import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MURADataset(Dataset):
    def __init__(self, df, transform=None, return_label=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.return_label = return_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["full_path"]).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        if self.return_label:
            return image, row["label"]

        return image


def load_mura_metadata(mura_root="data/MURA-v1.1"):
    train_csv = os.path.join(mura_root, "train_image_paths.csv")
    valid_csv = os.path.join(mura_root, "valid_image_paths.csv")

    train_df = pd.read_csv(train_csv, header=None, names=["path"])
    valid_df = pd.read_csv(valid_csv, header=None, names=["path"])

    train_df = train_df[train_df["path"].str.contains("XR_WRIST")].copy()
    valid_df = valid_df[valid_df["path"].str.contains("XR_WRIST")].copy()

    train_df["label"] = train_df["path"].apply(lambda x: 1 if "positive" in x else 0)
    valid_df["label"] = valid_df["path"].apply(lambda x: 1 if "positive" in x else 0)

    train_df["full_path"] = train_df["path"].apply(lambda x: os.path.join("data", x))
    valid_df["full_path"] = valid_df["path"].apply(lambda x: os.path.join("data", x))

    return train_df, valid_df


def get_mura_loaders(batch_size=16, img_size=256, num_workers=0):
    train_df, valid_df = load_mura_metadata()

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    recon_train_df = train_df[train_df["label"] == 0].copy()
    recon_val_df = valid_df[valid_df["label"] == 0].copy()

    recon_train_dataset = MURADataset(recon_train_df, transform=train_transform, return_label=False)
    recon_val_dataset = MURADataset(recon_val_df, transform=valid_transform, return_label=False)

    clf_train_dataset = MURADataset(train_df, transform=train_transform, return_label=True)
    clf_val_dataset = MURADataset(valid_df, transform=valid_transform, return_label=True)

    mura_recon_train_loader = DataLoader(
        recon_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    mura_recon_val_loader = DataLoader(
        recon_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    mura_clf_train_loader = DataLoader(
        clf_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    mura_clf_val_loader = DataLoader(
        clf_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return {
        "mura_recon_train_loader": mura_recon_train_loader,
        "mura_recon_val_loader": mura_recon_val_loader,
        "mura_clf_train_loader": mura_clf_train_loader,
        "mura_clf_val_loader": mura_clf_val_loader,
    }

def test_mura_loader():
    loaders = get_mura_loaders(batch_size=4, img_size=128, num_workers=0)

    for name, loader in loaders.items():
        print(f"\n{name}")
        batch = next(iter(loader))

        if isinstance(batch, list) or isinstance(batch, tuple):
            images, labels = batch
            print(images.shape, labels.shape)
        else:
            print(batch.shape)

test_mura_loader()