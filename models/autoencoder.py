import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=512, img_size=224):
        super().__init__()
        feat_size = img_size // 16  # 4 stride-2 convolutions
        flat_dim = 256 * feat_size * feat_size

        self.encoder = nn.Sequential(
            ConvBlock(1, 32, stride=2),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 256, stride=2),
            nn.Flatten(),
            nn.Linear(flat_dim, latent_dim),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, flat_dim),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, feat_size, feat_size)),
            ConvTransposeBlock(256, 128),
            ConvTransposeBlock(128, 64),
            ConvTransposeBlock(64, 32),
            ConvTransposeBlock(32, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruct(self, x):
        return self.forward(x)


if __name__ == "__main__":
    model = Autoencoder(img_size=224)
    x = torch.randn(2, 1, 224, 224)
    out = model(x)
    print("Input:", x.shape, "Output:", out.shape)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
