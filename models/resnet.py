import os
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


def build_resnet18(
    num_classes: int = 2,
    checkpoint: Optional[str] = None,
    map_location: str = "cpu",
) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if checkpoint:
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        state = torch.load(checkpoint, map_location=map_location)
        state_dict = state.get("model_state", state)
        model.load_state_dict(state_dict, strict=False)

    return model


def build_resnet34(
    num_classes: int = 2,
    checkpoint: Optional[str] = None,
    map_location: str = "cpu",
) -> nn.Module:
    model = models.resnet34(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if checkpoint:
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        state = torch.load(checkpoint, map_location=map_location)
        state_dict = state.get("model_state", state)
        model.load_state_dict(state_dict, strict=False)

    return model