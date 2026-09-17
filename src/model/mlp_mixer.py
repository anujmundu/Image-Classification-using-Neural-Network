import timm
import torch.nn as nn

def build_mlp_mixer(num_classes: int = 10, model_name: str = 'mixer_b16_224', **kwargs) -> nn.Module:
    """Create an MLP‑Mixer model via timm.

    Args:
        num_classes: Number of output classes.
        model_name: Specific MLP‑Mixer variant name recognized by timm.
        **kwargs: Additional kwargs passed to ``timm.create_model``.
    """
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes, **kwargs)
    return model
