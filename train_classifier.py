import os
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.graz_loader import get_graz_loaders
from data.mura_loader import get_mura_loaders
from models.classifier import build_classifier

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
BATCH_SIZE = 32
IMG_SIZE = 224
LR = 1e-4
NUM_WORKERS = 4
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs/classifier"
DATASET = "GRAZ"  # "graz" or "mura"


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
    all_preds = (all_probs >= 0.5).astype(int)

    auroc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)

    return total_loss / total, correct / total, auroc, f1, precision, recall


def get_loaders():
    if DATASET == "graz":
        return get_graz_loaders(batch_size=BATCH_SIZE, img_size=IMG_SIZE, num_workers=NUM_WORKERS)
    else:
        loaders = get_mura_loaders(batch_size=BATCH_SIZE, img_size=IMG_SIZE, num_workers=NUM_WORKERS)
        return loaders["mura_clf_train_loader"], loaders["mura_clf_val_loader"]


def save_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    _, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs, history["train_loss"], label="Train")
    axes[0, 0].plot(epochs, history["val_loss"], label="Val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, history["train_acc"], label="Train")
    axes[0, 1].plot(epochs, history["val_acc"], label="Val")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, history["val_auroc"], color="darkorange", label="AUROC")
    axes[1, 0].set_title("AUROC")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, history["val_f1"], label="F1")
    axes[1, 1].plot(epochs, history["val_precision"], label="Precision")
    axes[1, 1].plot(epochs, history["val_recall"], label="Recall")
    axes[1, 1].set_title("F1 / Precision / Recall")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"  Saved training curves to {output_dir}/training_curves.png")


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_loader, val_loader = get_loaders()
    model = build_classifier(num_classes=2, pretrained=True).to(DEVICE)

    labels = torch.tensor(train_loader.dataset.df["label"].values, dtype=torch.float32)
    n_neg = (labels == 0).sum()
    n_pos = (labels == 1).sum()
    class_weights = torch.tensor([n_pos / (n_neg + n_pos), n_neg / (n_neg + n_pos)]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    best_auroc = 0.0
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_auroc": [], "val_f1": [], "val_precision": [], "val_recall": [],
    }

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_auroc, val_f1, val_prec, val_rec = evaluate(model, val_loader, criterion)
        scheduler.step(val_auroc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_auroc"].append(val_auroc)
        history["val_f1"].append(val_f1)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_rec)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"Val loss {val_loss:.4f} acc {val_acc:.3f} "
            f"AUROC {val_auroc:.3f} F1 {val_f1:.3f} P {val_prec:.3f} R {val_rec:.3f}"
        )

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            path = os.path.join(CHECKPOINT_DIR, f"classifier_{DATASET}_best.pt")
            torch.save(model.state_dict(), path)
            print(f"  Saved best model (AUROC {best_auroc:.3f})")

    save_plots(history, OUTPUT_DIR)
    print(f"\nTraining complete. Best AUROC: {best_auroc:.3f}")


if __name__ == "__main__":
    main()
