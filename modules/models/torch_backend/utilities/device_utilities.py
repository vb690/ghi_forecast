import numpy as np

import torch


def infer_available_device():
    """Infer if mps device can be used or needs to fall back on cpu"""
    available_devices = ["cpu"]
    if torch.backends.cudnn.is_available():
        available_devices.append("cudnn")
    return available_devices


def move_to_device(target_tensor, device):
    """Move a tensor to a given device."""
    if isinstance(target_tensor, np.ndarray):
        target_tensor = torch.tensor(target_tensor)
    target_tensor = target_tensor.float().to(device)
    return target_tensor
