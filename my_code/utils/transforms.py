import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import v2

def sar_transform(resize_size: int = 512, triple:bool=True):
    """
    Torch transform for LINEAR units SAR. with log transform (linear to dB amplitude units)
    """
    transforms_list = [
        transforms.Lambda(ensure_hwc), 
        # transforms.Lambda(shift_min0), # if min<0, shift everything to min==0 ### doesnt work now but OK if ensure positive pixel values
        transforms.Lambda(Log_transform),
        v2.ToImage(), # convert to 
        v2.Resize((resize_size, resize_size), v2.InterpolationMode.BILINEAR), 
        v2.ToDtype(torch.float32, scale=False), 
        transforms.Lambda(normalize_tensor), # normalize 0-1
    ]
    if triple:
        transforms_list.append(transforms.Lambda(triple_channels))

    composed = transforms.Compose(transforms_list)
    if triple:
        composed.name = f'sar_transform_{resize_size}_triple'
    else:
        composed.name = f'sar_transform_{resize_size}_no_triple'
    return composed


### utils func
def Log_transform(x):
    if isinstance(x, torch.Tensor):
        return torch.log1p(x) #robust to log(0)
        # return torch.log10(x)
    else:
        return torch.from_numpy(np.log1p(x))

def normalize_tensor(x):
    """Normalize by max value (avoid division by zero)."""
    return x / (x.max() + 1e-6)

def triple_channels(x: torch.Tensor) -> torch.Tensor:
    """Ensure the image has 3 channels (RGB)."""
    return x if x.shape[0] == 3 else x.repeat(3, 1, 1)

def shift_min0(x: torch.Tensor) -> torch.Tensor:
    min_x = x.min()
    return x - min_x if min_x < 0 else x

def ensure_hwc(x):
    """
    Ensure image is in (H, W, C) format for v2.ToImage().
    Accepts NumPy arrays or torch tensors.
    """
    # NumPy array
    if isinstance(x, np.ndarray):
        if x.ndim == 3 and x.shape[0] in range(1,8): # 8 channels max
            # (C, H, W) -> (H, W, C)
            return np.moveaxis(x, 0, -1)
        return x

    # Torch tensor
    if torch.is_tensor(x):
        if x.ndim == 3 and x.shape[0] in (1, 3):
            return x.permute(1, 2, 0)
        return x

    return x


### Below: not proof-checked

def dinov3_imagenet_transform(resize_size: int = 512):
    """
    SAR transformations for DINOv3 model. 
    https://github.com/facebookresearch/dinov3?tab=readme-ov-file#image-transforms
    """
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_image = v2.ToImage()
    apply_log = Log_transform()
    to_float = v2.ToDtype(torch.float32, scale=True)
    triple = triple_channels()
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_image, resize, apply_log, to_float, triple, normalize])


def dinov3_sat_transform(resize_size: int = 512):
    """
    SAR transformations for DINOv3 model. 
    https://github.com/facebookresearch/dinov3?tab=readme-ov-file#image-transforms
    """
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_image = v2.ToImage()
    apply_log = Log_transform()
    to_float = v2.ToDtype(torch.float32, scale=True)
    triple = triple_channels()
    normalize = v2.Normalize(
        mean=(0.430, 0.411, 0.296),
        std=(0.213, 0.156, 0.143),
    )
    return v2.Compose([to_image, resize, apply_log, to_float, triple, normalize])

    