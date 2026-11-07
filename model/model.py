import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def get_model(classes=10, pretrained=True):
    if pretrained:
        weights = MobileNet_V2_Weights.DEFAULT
        model = mobilenet_v2(weights=weights)
    else: model = mobilenet_v2(weights=None)

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, classes)

    return model