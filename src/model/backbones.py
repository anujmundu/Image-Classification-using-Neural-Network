import torch
import torch.nn as nn
from typing import Literal

# Import individual backbone constructors
from .rnn_lstm import build_rnn, build_lstm
from .capsnet import build_capsnet
from .gnn import build_gnn
from .autoencoder import build_autoencoder
from .mlp_mixer import build_mlp_mixer

# Existing CNN/ViT backbones (import from torchvision / timm)
import torchvision.models as models
import timm

def build_backbone(
    backbone: Literal[
        'resnet101',
        'efficientnet_b4',
        'vgg16',
        'alexnet',
        'vit_small',
        'vit_base',
        'vit_large',
        'rnn',
        'lstm',
        'capsnet',
        'gnn',
        'autoencoder',
        'mlp_mixer'
    ],
    num_classes: int = 10,
    **kwargs,
) -> nn.Module:
    """Factory that returns a model instance for the requested backbone.

    Args:
        backbone: Name of the backbone architecture.
        num_classes: Number of output classes for the final classifier.
        **kwargs: Additional hyper‑parameters forwarded to specific builders.
    """
    # Classic CNN backbones
    if backbone == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if backbone == 'efficientnet_b4':
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if backbone == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    if backbone == 'alexnet':
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    # Modern Expanded Architectures
    if backbone == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    if backbone == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT, memory_efficient=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    if backbone == 'convnext_tiny':
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model
    if backbone == 'swin_t' or backbone == 'swin_tiny':
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    if backbone == 'shufflenet_v2_x1_0':
        model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if backbone == 'resnext50_32x4d':
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # Vision Transformers via timm
    if backbone.startswith('vit'):
        vit_map = {
            'vit_small': 'vit_small_patch16_224',
            'vit_base': 'vit_base_patch16_224',
            'vit_large': 'vit_large_patch16_224',
        }
        model_name = vit_map[backbone]
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        return model
    # Sequential models
    if backbone == 'rnn':
        return build_rnn(num_classes=num_classes, **kwargs)
    if backbone == 'lstm':
        return build_lstm(num_classes=num_classes, **kwargs)
    # Capsule Network
    if backbone == 'capsnet':
        return build_capsnet(num_classes=num_classes, **kwargs)
    # Graph Neural Network
    if backbone == 'gnn':
        return build_gnn(num_classes=num_classes, **kwargs)
    # Autoencoder (encoder part used as feature extractor, plus classifier head)
    if backbone == 'autoencoder':
        return build_autoencoder(num_classes=num_classes, **kwargs)
    # MLP-Mixer
    if backbone == 'mlp_mixer':
        return build_mlp_mixer(num_classes=num_classes, **kwargs)
    raise ValueError(f'Unsupported backbone: {backbone}')
