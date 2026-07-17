import cv2
import numpy as np

def apply_gaussian_blur(img, ksize, sigma):
    """Áp dụng bộ lọc Gaussian (Gaussian Blur)"""
    kernel_1d = cv2.getGaussianKernel(ksize, sigma)
    kernel = np.outer(kernel_1d, kernel_1d).astype(np.float32)
    kernel_name = f"Gaussian Blur {ksize}x{ksize} (sigma={sigma if sigma > 0 else 'Auto'})"
    img_filtered = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return img_filtered, kernel, kernel_name
