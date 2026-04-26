import os
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import save_image

from data.graz_loader import get_graz_loaders
from data.mura_loader import get_mura_loaders
from models.classifier import build_classifier
from models.autoencoder import Autoencoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Classifier

@torch.no_grad()
def eval_classifier(checkpoint, dataset="graz", batch_size=16, img_size=256):
    print(f"\n=== Classifier evaluation ({dataset}) ===")

    if dataset == "graz":
        _, val_loader = get_graz_loaders(batch_size=batch_size, img_size=img_size)
    else:
        loaders = get_mura_loaders(batch_size=batch_size, img_size=img_size)
        val_loader = loaders["mura_clf_val_loader"]

    model = build_classifier(num_classes=2, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    all_preds, all_probs, all_labels = [], [], []

    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(1)

        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    print(f"  Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  AUROC     : {roc_auc_score(y_true, y_prob):.4f}")
    print(f"  Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1        : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Confusion matrix:\n{confusion_matrix(y_true, y_pred)}")


# Autoencoder

@torch.no_grad()
def eval_autoencoder(checkpoint, batch_size=16, img_size=256, n_samples=16,
                     output_dir="outputs/eval_autoencoder"):
    print("\n=== Autoencoder evaluation ===")
    os.makedirs(output_dir, exist_ok=True)

    loaders = get_mura_loaders(batch_size=batch_size, img_size=img_size)
    val_loader = loaders["mura_recon_val_loader"]

    model = Autoencoder().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    mse_list, ssim_list = [], []
    saved = 0

    for images in val_loader:
        images = images.to(DEVICE)
        recon = model(images)

        for i in range(images.size(0)):
            orig = images[i].cpu().numpy().squeeze()
            rec = recon[i].cpu().numpy().squeeze()

            orig_u8 = ((orig + 1) / 2 * 255).clip(0, 255)
            rec_u8 = ((rec + 1) / 2 * 255).clip(0, 255)

            mse_list.append(np.mean((orig_u8 - rec_u8) ** 2))
            ssim_list.append(ssim(orig_u8, rec_u8, data_range=255))

        if saved < n_samples:
            n = min(n_samples - saved, images.size(0))
            comparison = torch.cat([images[:n], recon[:n]], dim=0)
            save_image(comparison, os.path.join(output_dir, f"recon_{saved}.png"),
                       nrow=n, normalize=True)
            saved += n

    print(f"  MSE  (mean): {np.mean(mse_list):.2f}")
    print(f"  SSIM (mean): {np.mean(ssim_list):.4f}")


# Anomaly maps

@torch.no_grad()
def compute_anomaly_maps(checkpoint, dataset="graz", batch_size=8, img_size=256,
                         output_dir="outputs/anomaly_maps", n_batches=3):
    print("\n=== Anomaly maps ===")
    os.makedirs(output_dir, exist_ok=True)

    if dataset == "graz":
        _, val_loader = get_graz_loaders(batch_size=batch_size, img_size=img_size)
    else:
        loaders = get_mura_loaders(batch_size=batch_size, img_size=img_size)
        val_loader = loaders["mura_clf_val_loader"]

    model = Autoencoder().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    for batch_idx, (images, labels) in enumerate(val_loader):
        if batch_idx >= n_batches:
            break

        images = images.to(DEVICE)
        recon = model(images)
        diff = (images - recon).abs()

        for i in range(images.size(0)):
            label = labels[i].item()
            prefix = os.path.join(output_dir, f"b{batch_idx}_s{i}_label{label}")
            save_image(images[i], f"{prefix}_orig.png", normalize=True)
            save_image(recon[i], f"{prefix}_recon.png", normalize=True)
            save_image(diff[i], f"{prefix}_diff.png", normalize=True)

    print(f"  Saved maps to {output_dir}/")


# Fracture localization

def _tensor_to_pil(t):
    """Convert a normalized [-1, 1] single-channel tensor to a grayscale PIL image."""
    arr = ((t.squeeze().cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _bounding_box(mask):
    """Return (x0, y0, x1, y1) of the tightest box around non-zero pixels, or None."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return int(x0), int(y0), int(x1), int(y1)


@torch.no_grad()
def localize_fractures(checkpoint, dataset="graz", batch_size=8, img_size=256,
                       output_dir="outputs/localization", threshold_pct=95, n_batches=3):
    print("\n=== Fracture localization ===")
    os.makedirs(output_dir, exist_ok=True)

    if dataset == "graz":
        _, val_loader = get_graz_loaders(batch_size=batch_size, img_size=img_size)
    else:
        loaders = get_mura_loaders(batch_size=batch_size, img_size=img_size)
        val_loader = loaders["mura_clf_val_loader"]

    model = Autoencoder().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    for batch_idx, (images, labels) in enumerate(val_loader):
        if batch_idx >= n_batches:
            break

        images = images.to(DEVICE)
        recon = model(images)
        diff = (images - recon).abs()

        for i in range(images.size(0)):
            label = labels[i].item()

            diff_np = diff[i].squeeze().cpu().numpy()
            threshold = np.percentile(diff_np, threshold_pct)
            mask = (diff_np >= threshold).astype(np.uint8)

            orig_pil = _tensor_to_pil(images[i])
            recon_pil = _tensor_to_pil(recon[i])

            diff_norm = (diff_np / diff_np.max() * 255).clip(0, 255).astype(np.uint8)
            diff_pil = Image.fromarray(diff_norm).convert("RGB")

            bbox_pil = orig_pil.copy()
            bbox = _bounding_box(mask)
            if bbox is not None:
                draw = ImageDraw.Draw(bbox_pil)
                draw.rectangle(bbox, outline="red", width=3)

            w, h = orig_pil.size
            composite = Image.new("RGB", (w * 4, h))
            composite.paste(orig_pil,   (0,     0))
            composite.paste(recon_pil,  (w,     0))
            composite.paste(diff_pil,   (w * 2, 0))
            composite.paste(bbox_pil,   (w * 3, 0))

            fname = os.path.join(output_dir, f"b{batch_idx}_s{i}_label{label}.png")
            composite.save(fname)

    print(f"  Saved to {output_dir}/  (columns: original | reconstruction | diff | localization)")


# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", type=str, default=None,
                        help="Path to classifier checkpoint (.pt)")
    parser.add_argument("--autoencoder", type=str, default=None,
                        help="Path to autoencoder checkpoint (.pt)")
    parser.add_argument("--dataset", type=str, default="graz",
                        choices=["graz", "mura"])
    parser.add_argument("--anomaly_maps", action="store_true",
                        help="Compute and save anomaly maps")
    parser.add_argument("--localize", action="store_true",
                        help="Run fracture localization and save bounding box visualizations")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.classifier:
        eval_classifier(args.classifier, dataset=args.dataset)

    if args.autoencoder:
        eval_autoencoder(args.autoencoder)

    if args.anomaly_maps and args.autoencoder:
        compute_anomaly_maps(args.autoencoder, dataset=args.dataset)

    if args.localize and args.autoencoder:
        localize_fractures(args.autoencoder, dataset=args.dataset)
