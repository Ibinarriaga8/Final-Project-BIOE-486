import os
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.utils import save_image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.graz_loader import get_graz_recon_datasets, PIN_MEMORY
from data.mura_loader import get_mura_loaders
from models.autoencoder import Autoencoder

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
BATCH_SIZE = 32
IMG_SIZE = 224
LR = 1e-3
NUM_WORKERS = 4
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs/autoencoder"


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, total = 0.0, 0

    for images in loader:
        images = images.to(DEVICE)

        optimizer.zero_grad()
        recon = model(images)
        loss = criterion(recon, images)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

    return total_loss / total


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, total = 0.0, 0

    for images in loader:
        images = images.to(DEVICE)
        recon = model(images)
        loss = criterion(recon, images)
        total_loss += loss.item() * images.size(0)
        total += images.size(0)

    return total_loss / total


@torch.no_grad()
def save_reconstructions(model, loader, epoch, n=8):
    model.eval()
    images = next(iter(loader))[:n].to(DEVICE)
    recon = model(images)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    comparison = torch.cat([images, recon], dim=0)
    path = os.path.join(OUTPUT_DIR, f"recon_epoch{epoch:03d}.png")
    save_image(comparison, path, nrow=n, normalize=True, value_range=(-1, 1))


@torch.no_grad()
def compute_val_metrics(model, loader):
    model.eval()
    ssim_scores, psnr_scores = [], []

    for images in loader:
        images = images.to(DEVICE)
        recon = model(images)
        for i in range(images.size(0)):
            orig = ((images[i].squeeze().cpu().numpy() + 1) / 2 * 255).clip(0, 255)
            rec = ((recon[i].squeeze().cpu().numpy() + 1) / 2 * 255).clip(0, 255)
            ssim_scores.append(ssim(orig, rec, data_range=255))
            mse = np.mean((orig - rec) ** 2)
            psnr_scores.append(20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100.0)

    return float(np.mean(ssim_scores)), float(np.mean(psnr_scores))


def save_plots(train_losses, val_losses, val_ssims, output_dir):
    epochs = range(1, len(train_losses) + 1)
    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_losses, label="Train")
    axes[0].plot(epochs, val_losses, label="Val")
    axes[0].set_title("Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()

    axes[1].plot(epochs, val_ssims, color="steelblue", label="Val SSIM")
    axes[1].set_title("Reconstruction Quality (SSIM)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("SSIM")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"  Saved training curves to {output_dir}/training_curves.png")


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mura_loaders = get_mura_loaders(batch_size=BATCH_SIZE, img_size=IMG_SIZE)
    mura_train_ds = mura_loaders["mura_recon_train_loader"].dataset
    mura_val_ds = mura_loaders["mura_recon_val_loader"].dataset

    graz_train_ds, graz_val_ds = get_graz_recon_datasets(img_size=IMG_SIZE)


    train_dataset = ConcatDataset([mura_train_ds, graz_train_ds])
    val_dataset = ConcatDataset([mura_val_ds, graz_val_ds])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=NUM_WORKERS > 0)

    print(f"Train: {len(train_dataset)} normal images (MURA + GRAZ)")
    print(f"Val:   {len(val_dataset)} normal images (MURA + GRAZ)")

    model = Autoencoder(img_size=IMG_SIZE).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    best_loss = float("inf")
    train_losses, val_losses, val_ssims = [], [], []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate(model, val_loader, criterion)
        val_ssim, val_psnr = compute_val_metrics(model, val_loader)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ssims.append(val_ssim)

        print(f"Epoch {epoch:02d}/{EPOCHS} | Train loss {train_loss:.6f} | Val loss {val_loss:.6f} | SSIM {val_ssim:.4f} | PSNR {val_psnr:.2f} dB")

        if epoch % 10 == 0:
            save_reconstructions(model, val_loader, epoch)

        if val_loss < best_loss:
            best_loss = val_loss
            path = os.path.join(CHECKPOINT_DIR, "autoencoder_best.pt")
            torch.save(model.state_dict(), path)
            print(f"  Saved best model (loss {best_loss:.6f})")

    save_plots(train_losses, val_losses, val_ssims, OUTPUT_DIR)
    print(f"\nTraining complete. Best val loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
