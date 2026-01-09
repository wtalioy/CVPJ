import importlib
import torch.nn as nn
from typing import Tuple

IMG_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
def is_image(fname: str) -> bool:
    return fname.lower().endswith(IMG_EXTS)


def build_model(model_str: str, num_classes: int, map_location: str = "cpu") -> nn.Module:
    if ":" in model_str:
        module_name, fn_name = model_str.split(":", 1)
    else:
        module_name, fn_name = "models", f"build_{model_str}"

    module = importlib.import_module(module_name)
    builder = getattr(module, fn_name)

    return builder(num_classes=num_classes, map_location=map_location)

