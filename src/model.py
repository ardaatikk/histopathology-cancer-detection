import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class CancerDetectionModel(nn.Module):
    """ResNet-18 based binary classifier for histopathology images."""

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # Remove the original ImageNet classification layer.
        self.resnet18 = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.fc1 = nn.Linear(512, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.resnet18(x)
        x = x.flatten(1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x