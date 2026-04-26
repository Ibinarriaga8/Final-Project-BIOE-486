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
    def __init__(self, latent_dim=512):
        super().__init__()

        # encoder: 1x256x256 → latent_dim
        self.encoder = nn.Sequential(
            ConvBlock(1, 32, stride=2),    # 128x128
            ConvBlock(32, 64, stride=2),   # 64x64
            ConvBlock(64, 128, stride=2),  # 32x32
            ConvBlock(128, 256, stride=2), # 16x16
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, latent_dim),
            nn.ReLU(inplace=True),
        )

        # decoder: latent_dim → 1x256x256
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 16 * 16),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 16, 16)),
            ConvTransposeBlock(256, 128),  # 32x32
            ConvTransposeBlock(128, 64),   # 64x64
            ConvTransposeBlock(64, 32),    # 128x128
            ConvTransposeBlock(32, 1),     # 256x256
            nn.Tanh(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruct(self, x):
        return self.forward(x)


if __name__ == "__main__":
    model = Autoencoder()
    x = torch.randn(2, 1, 256, 256)
    out = model(x)
    print("Input:", x.shape, "Output:", out.shape)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
