import cv2
import numpy as np

def apply_sobel_filter(img, ksize, direction):
    """Áp dụng bộ lọc Sobel (Phát hiện biên bậc 1)"""
    kernel = None
    
    if direction == '1' or direction == 'X' or direction == 'x':
        if ksize == 3:
            kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=np.float32)
        kernel_name = f"Sobel X (ksize={ksize})"
        sobel_val = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
        img_filtered = cv2.convertScaleAbs(sobel_val)
        
    elif direction == '2' or direction == 'Y' or direction == 'y':
        if ksize == 3:
            kernel = np.array([[-1, -2, -1],
                               [ 0,  0,  0],
                               [ 1,  2,  1]], dtype=np.float32)
        kernel_name = f"Sobel Y (ksize={ksize})"
        sobel_val = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
        img_filtered = cv2.convertScaleAbs(sobel_val)
        
    else:
        if ksize == 3:
            # Dùng Sobel X làm đại diện cho ma trận hiển thị
            kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=np.float32)
        kernel_name = f"Sobel Magnitude (ksize={ksize})"
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
        sobel_magnitude = cv2.magnitude(sobelx, sobely)
        img_filtered = cv2.convertScaleAbs(sobel_magnitude)
        
    return img_filtered, kernel, kernel_name
