import cv2
import numpy as np

def apply_laplacian_filter(img, ksize):
    """Áp dụng bộ lọc Laplacian (Phát hiện biên bậc 2)"""
    kernel_name = f"Laplacian (ksize={ksize})"
    
    # Tạo ma trận tương đương để hiển thị
    if ksize == 1:
        kernel = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=np.float32)
    elif ksize == 3:
        kernel = np.array([[2, 0, 2],
                           [0, -8, 0],
                           [2, 0, 2]], dtype=np.float32)
    else:
        kernel = None  # Không sinh ma trận thủ công cho size lớn hơn
        print(f"\n  Laplacian ksize={ksize} (kernel lớn, không hiển thị ma trận).\n")
        
    res_lap = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
    img_filtered = cv2.convertScaleAbs(res_lap)
    return img_filtered, kernel, kernel_name
