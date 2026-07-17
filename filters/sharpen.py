import cv2
import numpy as np

def apply_sharpen_filter(img):
    """Áp dụng bộ lọc làm sắc nét (Sharpen)"""
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]], dtype=np.float32)
    kernel_name = "Sharpen 3x3"
    img_filtered = cv2.filter2D(img, -1, kernel)
    return img_filtered, kernel, kernel_name
