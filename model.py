import torch.nn as nn
from torchvision.models import efficientnet_b0, resnet50

class ChestXrayClassifier(nn.Module):
    def __init__(self, num_classes, backbone="efficientnet"):
        super().__init__()

        if backbone == "efficientnet":
            self.model = efficientnet_b0(weights="DEFAULT")
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_features, num_classes)

        elif backbone == "resnet":
            self.model = resnet50(weights=None)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)

        else:
            raise ValueError("Invalid backbone")

    def forward(self, x):
        return self.model(x)

