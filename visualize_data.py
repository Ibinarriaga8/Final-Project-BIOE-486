"""
Generates presentation-ready figures:
  outputs/figures/graz_samples.png     — grid of GRAZ normal vs fracture X-rays
  outputs/figures/mura_samples.png     — grid of MURA normal vs abnormal X-rays
  outputs/figures/dataset_stats.png    — class distribution bar charts
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from data.graz_loader import load_graz_metadata, GRAZDataset
from data.mura_loader import load_mura_metadata
from torchvision import transforms

OUTPUT_DIR = "outputs/figures"
IMG_SIZE = 256
N_COLS = 4


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])


def tensor_to_np(t):
    return t.squeeze().numpy()


def save_sample_grid(dataset, labels, title, filepath, n_normal=N_COLS, n_abnormal=N_COLS):
    normal_idx = [i for i, l in enumerate(labels) if l == 0][:n_normal]
    abnormal_idx = [i for i, l in enumerate(labels) if l == 1][:n_abnormal]

    fig, axes = plt.subplots(2, N_COLS, figsize=(N_COLS * 3, 7))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for col, idx in enumerate(normal_idx):
        img, _ = dataset[idx]
        axes[0, col].imshow(tensor_to_np(img), cmap="gray")
        axes[0, col].axis("off")
    axes[0, 0].set_ylabel("Normal", fontsize=11, rotation=90, labelpad=10)

    for col, idx in enumerate(abnormal_idx):
        img, _ = dataset[idx]
        axes[1, col].imshow(tensor_to_np(img), cmap="gray")
        axes[1, col].axis("off")
    axes[1, 0].set_ylabel("Fracture / Abnormal", fontsize=11, rotation=90, labelpad=10)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filepath}")


def save_distribution_plot(graz_df, mura_train_df, mura_val_df, filepath):
    _, axes = plt.subplots(1, 3, figsize=(14, 4))

    # GRAZ
    counts = graz_df["label"].value_counts().sort_index()
    axes[0].bar(["Normal", "Fracture"], counts.values, color=["steelblue", "tomato"])
    axes[0].set_title("GRAZ — Class Distribution")
    axes[0].set_ylabel("Images")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 30, str(v), ha="center", fontsize=10)

    # MURA train
    counts = mura_train_df["label"].value_counts().sort_index()
    axes[1].bar(["Normal", "Abnormal"], counts.values, color=["steelblue", "tomato"])
    axes[1].set_title("MURA Train — Class Distribution")
    axes[1].set_ylabel("Images")
    for i, v in enumerate(counts.values):
        axes[1].text(i, v + 10, str(v), ha="center", fontsize=10)

    # MURA val
    counts = mura_val_df["label"].value_counts().sort_index()
    axes[2].bar(["Normal", "Abnormal"], counts.values, color=["steelblue", "tomato"])
    axes[2].set_title("MURA Val — Class Distribution")
    axes[2].set_ylabel("Images")
    for i, v in enumerate(counts.values):
        axes[2].text(i, v + 5, str(v), ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filepath}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    transform = get_transform()

    print("Loading GRAZ metadata...")
    graz_df = load_graz_metadata()
    graz_dataset = GRAZDataset(graz_df, transform=transform, return_label=True)
    graz_labels = graz_df["label"].tolist()
    save_sample_grid(
        graz_dataset, graz_labels,
        "GRAZ — Wrist X-rays (Normal vs Fracture)",
        os.path.join(OUTPUT_DIR, "graz_samples.png"),
    )

    print("Loading MURA metadata...")
    mura_train_df, mura_val_df = load_mura_metadata()
    from data.mura_loader import MURADataset
    mura_dataset = MURADataset(mura_train_df, transform=transform, return_label=True)
    mura_labels = mura_train_df["label"].tolist()
    save_sample_grid(
        mura_dataset, mura_labels,
        "MURA — Wrist X-rays (Normal vs Abnormal)",
        os.path.join(OUTPUT_DIR, "mura_samples.png"),
    )

    print("Saving class distribution chart...")
    save_distribution_plot(
        graz_df, mura_train_df, mura_val_df,
        os.path.join(OUTPUT_DIR, "dataset_stats.png"),
    )

    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
