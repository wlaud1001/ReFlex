import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
from skimage.filters import threshold_otsu


def gaussian_blur(image, kernel_size=7, sigma=2):
    """
    Apply Gaussian blur to a binary mask image.
    
    Args:
        image (torch.Tensor): Input binary mask (1x1xHxW or HxW) as a PyTorch tensor.
        kernel_size (int): Size of the Gaussian kernel. Should be odd.
        sigma (float): Standard deviation of the Gaussian kernel.
    
    Returns:
        torch.Tensor: Blurred mask image.
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Generate Gaussian kernel
    x = torch.arange(kernel_size, device=image.device, dtype=image.dtype) - kernel_size // 2
    gaussian_1d = torch.exp(-(x**2) / (2 * sigma**2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    gaussian_kernel = gaussian_1d[:, None] * gaussian_1d[None, :]
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()  # Normalize

    # Reshape to fit convolution: (out_channels, in_channels, kH, kW)
    gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)

    # Ensure image is 4D (BxCxHxW)
    if image.ndim == 2:  # HxW
        image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    elif image.ndim == 3:  # CxHxW
        image = image.unsqueeze(0)  # Add batch dimension
    
    # Convolve image with Gaussian kernel
    blurred_image = F.conv2d(image, gaussian_kernel, padding=kernel_size // 2)
    
    return blurred_image.squeeze()  # Remove extra dimensions

def mask_interpolate(mask, size=128):
    mask = torch.tensor(mask)
    mask = F.interpolate(mask[None, None, ...], size, mode='bicubic')
    mask = mask.squeeze()
    return mask 

def get_mask(ca, ca_index, gb_kernel=11, gb_sigma=2, dilation=1, nbins=64):
    if ca is None:
        return None
    else:
        ca = ca[0].mean(0)
        token_ca = ca[..., ca_index].mean(dim=-1).reshape(64, 64)
        token_ca = gaussian_blur(token_ca, kernel_size=gb_kernel, sigma=gb_sigma)
        token_ca = mask_interpolate(token_ca, size=1024)
        thres = threshold_otsu(token_ca.float().cpu().numpy(), nbins=nbins)
        mask = token_ca > thres
        mask = mask_interpolate(mask.to(ca.dtype), 128)
        if dilation:
            mask = binary_dilation(mask.float().cpu().numpy(), iterations=dilation)
        mask = torch.tensor(mask, device=ca.device, dtype=ca.dtype)
        return mask