"""Utility functions for device management and deterministic behavior."""

import torch
import random
import numpy as np
import os
from typing import Optional, Union


def get_device(device: str = "auto") -> torch.device:
    """
    Get the best available device with fallback chain: CUDA → MPS → CPU.
    
    Args:
        device: Device preference ("auto", "cuda", "mps", "cpu")
        
    Returns:
        torch.device: The selected device
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def set_deterministic(seed: int = 42, device: Optional[torch.device] = None) -> None:
    """
    Set deterministic behavior for reproducibility.
    
    Args:
        seed: Random seed
        device: PyTorch device (optional)
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    
    # Set CUDA random seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set MPS random seed if available (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # Set environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def get_device_info(device: torch.device) -> dict:
    """
    Get information about the current device.
    
    Args:
        device: PyTorch device
        
    Returns:
        dict: Device information
    """
    info = {
        "device": str(device),
        "type": device.type,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
    }
    
    if device.type == "cuda":
        info.update({
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_current_device": torch.cuda.current_device(),
            "cuda_device_name": torch.cuda.get_device_name(),
            "cuda_memory_allocated": torch.cuda.memory_allocated(),
            "cuda_memory_reserved": torch.cuda.memory_reserved(),
        })
    
    return info


def print_device_info(device: torch.device) -> None:
    """Print device information in a formatted way."""
    info = get_device_info(device)
    print(f"Using device: {info['device']}")
    
    if device.type == "cuda":
        print(f"CUDA device: {info['cuda_device_name']}")
        print(f"Memory allocated: {info['cuda_memory_allocated'] / 1024**2:.2f} MB")
        print(f"Memory reserved: {info['cuda_memory_reserved'] / 1024**2:.2f} MB")
    elif device.type == "mps":
        print("Using Apple Silicon GPU (MPS)")
    else:
        print("Using CPU")
