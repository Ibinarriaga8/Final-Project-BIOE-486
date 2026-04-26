import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.utils import save_image

from data.graz_loader import get_graz_recon_datasets, PIN_MEMORY
from data.mura_loader import get_mura_loaders
from models.autoencoder import Autoencoder

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 32
IMG_SIZE = 256
LR = 1e-3
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
    save_image(comparison, path, nrow=n, normalize=True)


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    mura_loaders = get_mura_loaders(batch_size=BATCH_SIZE, img_size=IMG_SIZE)
    mura_train_ds = mura_loaders["mura_recon_train_loader"].dataset
    mura_val_ds = mura_loaders["mura_recon_val_loader"].dataset

    graz_train_ds, graz_val_ds = get_graz_recon_datasets(img_size=IMG_SIZE)

    train_dataset = ConcatDataset([mura_train_ds, graz_train_ds])
    val_dataset = ConcatDataset([mura_val_ds, graz_val_ds])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY)

    print(f"Train: {len(train_dataset)} normal images (MURA + GRAZ)")
    print(f"Val:   {len(val_dataset)} normal images (MURA + GRAZ)")

    model = Autoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate(model, val_loader, criterion)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{EPOCHS} | Train loss {train_loss:.6f} | Val loss {val_loss:.6f}")

        if epoch % 10 == 0:
            save_reconstructions(model, val_loader, epoch)

        if val_loss < best_loss:
            best_loss = val_loss
            path = os.path.join(CHECKPOINT_DIR, "autoencoder_best.pt")
            torch.save(model.state_dict(), path)
            print(f"  Saved best model (loss {best_loss:.6f})")

    print(f"\nTraining complete. Best val loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
