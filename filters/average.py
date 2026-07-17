import cv2
import numpy as np

def apply_average_blur(img, ksize):
    """Áp dụng bộ lọc trung bình (Average Blur)"""
    kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
    kernel_name = f"Average Blur {ksize}x{ksize}"
    img_filtered = cv2.filter2D(img, -1, kernel)
    return img_filtered, kernel, kernel_name
