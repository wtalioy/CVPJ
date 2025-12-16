"""Model registry for the project."""

from .specxnet import build_specxnet
from .resnet import build_resnet18, build_resnet34
from .ferretnet import build_ferretnet
from .artifactnet import build_artifactnet

__all__ = ["build_specxnet", "build_resnet18", "build_resnet34", "build_ferretnet", "build_artifactnet"]

