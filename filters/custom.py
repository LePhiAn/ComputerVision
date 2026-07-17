import cv2
import numpy as np

def apply_custom_kernel(img, kernel, kernel_name="Custom Kernel"):
    """Áp dụng bộ lọc với kernel tùy chọn người dùng nhập"""
    tong = np.sum(kernel)
    # Chống tràn màu cho các bộ lọc có hệ số âm (như Edge Detection, Sharpen)
    if tong <= 0 or np.any(kernel < 0):
        res_cv = cv2.filter2D(img, cv2.CV_64F, kernel)
        img_filtered = cv2.convertScaleAbs(res_cv)
    else:
        img_filtered = cv2.filter2D(img, -1, kernel)
    return img_filtered, kernel, kernel_name
