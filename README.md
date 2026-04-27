# BIOE 486 — Anomaly Detection in Musculoskeletal Radiographs

Final project for BIOE 486 (Applied Deep Learning for Biomedical Imaging).

Compares a supervised baseline classifier (ResNet-18) against an unsupervised convolutional autoencoder for anomaly detection on wrist/hand X-rays (GRAZ + MURA datasets).

## Setup

```bash
pip install -r requirements.txt
```

Datasets are downloaded automatically via `kagglehub` on first run. Requires a [Kaggle API key](https://www.kaggle.com/docs/api).

## Pipeline

### 1. Train the baseline classifier

```bash
python train_classifier.py
```

Trains ResNet-18 on GRAZ (normal vs. abnormal). Saves best model to `checkpoints/classifier_graz_best.pt` and curves to `outputs/classifier/`.

### 2. Train the autoencoder

```bash
python train_autoencoder.py
```

Trains a convolutional autoencoder on **normal images only** (MURA + GRAZ). Saves best model to `checkpoints/autoencoder_best.pt` and reconstruction samples to `outputs/autoencoder/`.

### 3. Evaluate

```bash
python evaluate.py
```

Computes AUROC, F1, precision and recall for both models. Results are saved to `outputs/`.

## Project structure

```
data/           # dataset loaders (GRAZ, MURA)
models/         # classifier and autoencoder architectures
checkpoints/    # saved model weights (git-ignored)
outputs/        # plots and reconstruction images (git-ignored)
train_classifier.py
train_autoencoder.py
evaluate.py
```
