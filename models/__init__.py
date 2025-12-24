"""Model registry for the project."""

from .specxnet import build_specxnet
from .ferretnet_ddfc import build_ferretnet_ddfc
from .resnet import build_resnet18, build_resnet34
from .ferretnet import build_ferretnet
from .artifactnet import build_artifactnet
from .ferret_v4 import build_ferret_v4

__all__ = ["build_specxnet", "build_ferretnet_ddfc", "build_resnet18", "build_resnet34", "build_ferretnet", "build_artifactnet", "build_ferret_v4"]


