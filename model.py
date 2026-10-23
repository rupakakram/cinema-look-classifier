"""
Cinema Look Classifier
Input: 224x224 RGB image
Output: 2 class scores [cinematic, normal]
"""

import torch
import torch.nn as nn

class CinemaClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super(CinemaClassifier, self).__init__()

        # Each Conv2d block: Conv -> ReLU -> MaxPool
        self.features = nn.Sequential(

            # Block 1 - Input layer receives 3 channels (RGB)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 224 -> 112

            # B;ock 2 - Hidden layer 1
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 112 -> 56

            # Block 3 - Hidden layer 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 56 -> 28

            # Block 4 - Hidden layer 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 28 -> 14
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), # Flatten the feature maps into a vector
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(128, num_classes), # Output layer: 2 nodes (cinematic, normal)
        )

    def forward(self, x):
        x = self.features(x) # Extract features
        x = self.classifier(x) # Classify based on features
        return x
        
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

if __name__ == "__main__":
    model = CinemaClassifier()
    count_parameters(model)

    dummy = torch.randn(4, 3, 224, 224) # Batch of 4 RGB images
    out = model(dummy)
    print(f"Output shape: {out.shape}") # Should be [4,2]
