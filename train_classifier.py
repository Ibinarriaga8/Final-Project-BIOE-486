import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import numpy as np

from data.graz_loader import get_graz_loaders
from data.mura_loader import get_mura_loaders
from models.classifier import build_classifier

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
BATCH_SIZE = 16
IMG_SIZE = 256
LR = 1e-4
CHECKPOINT_DIR = "checkpoints"
DATASET = "graz"  # "graz" or "mura"


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    auroc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0

    return total_loss / total, correct / total, auroc


def get_loaders():
    if DATASET == "graz":
        return get_graz_loaders(batch_size=BATCH_SIZE, img_size=IMG_SIZE)
    else:
        loaders = get_mura_loaders(batch_size=BATCH_SIZE, img_size=IMG_SIZE)
        return loaders["mura_clf_train_loader"], loaders["mura_clf_val_loader"]


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_loader, val_loader = get_loaders()
    model = build_classifier(num_classes=2, pretrained=True).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    best_auroc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_auroc = evaluate(model, val_loader, criterion)
        scheduler.step(val_auroc)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"Val loss {val_loss:.4f} acc {val_acc:.3f} AUROC {val_auroc:.3f}"
        )

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            path = os.path.join(CHECKPOINT_DIR, f"classifier_{DATASET}_best.pt")
            torch.save(model.state_dict(), path)
            print(f"  Saved best model (AUROC {best_auroc:.3f})")

    print(f"\nTraining complete. Best AUROC: {best_auroc:.3f}")


if __name__ == "__main__":
    main()
