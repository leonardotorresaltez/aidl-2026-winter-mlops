import torch
import torch.nn as nn


class MyModel(nn.Module):
    """
    Simple convolutional classifier for grayscale images.

    - Expects input tensors of shape (B, 1, H, W).
    - Uses adaptive pooling so H and W can vary.
    - Outputs raw logits of shape (B, num_classes).
    """

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            #nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm2d(128),
        )

        # Reduce spatial dims to (1,1) regardless of input size so Flatten
        # produces exactly `channels` features (here 64) per sample.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            #nn.Dropout(0.5),
            #nn.Linear(128, num_classes),
            nn.Linear(64, 128),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = self.pad = torch.nn.ConstantPad2d(2,0)

        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x