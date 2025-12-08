"""Model registry for the project."""

from .specxnet import build_specxnet
from .resnet import build_resnet18, build_resnet34

__all__ = ["build_specxnet", "build_resnet18", "build_resnet34"]

