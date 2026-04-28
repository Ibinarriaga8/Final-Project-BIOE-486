import torch
import torch.nn as nn
from torchvision import models


def build_classifier(num_classes=2, pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # adapt first conv to accept single-channel (grayscale) images
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


if __name__ == "__main__":
    model = build_classifier()
    x = torch.randn(2, 1, 256, 256)
    out = model(x)
    print("Output shape:", out.shape)
