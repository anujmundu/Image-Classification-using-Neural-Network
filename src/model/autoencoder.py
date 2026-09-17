import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 112x112
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 56x56
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 28 * 28, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 28 * 28),
            nn.Unflatten(1, (256, 28, 28)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.decoder(z)

def build_autoencoder(num_classes: int = 10, latent_dim: int = 128, **kwargs) -> nn.Module:
    """Autoencoder used as feature extractor + classifier head.
    The encoder output is fed to a linear classifier.
    """
    class AutoEncoderModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder(latent_dim)
            self.decoder = Decoder(latent_dim)
            self.classifier = nn.Linear(latent_dim, num_classes)
        def forward(self, x):
            z = self.encoder(x)
            logits = self.classifier(z)
            # For reconstruction tasks you can also return self.decoder(z)
            return logits
    return AutoEncoderModel()
