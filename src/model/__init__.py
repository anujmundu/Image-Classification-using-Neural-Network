import torch.nn as nn
import torchvision.models as models
from .backbones import build_backbone


def build_model(num_classes: int, backbone: str = 'resnet101', **kwargs) -> nn.Module:
    """Factory wrapper that forwards to the extended backbones factory.

    Args:
        num_classes: Number of output classes.
        backbone: Name of the backbone architecture.
        **kwargs: Additional arguments passed to specific builders.

    Returns:
        A torch.nn.Module ready for training/inference.
    """
    if backbone == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if backbone == 'efficientnet_b4':
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    if backbone == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        return model
    # For other backbones delegate to backbones factory
    return build_backbone(backbone, num_classes=num_classes, **kwargs)

