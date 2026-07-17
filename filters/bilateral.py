import cv2

def apply_bilateral_filter(img, d, sigma_color, sigma_space):
    """Áp dụng bộ lọc song biên / 2 chiều (Bilateral Filter)"""
    kernel_name = f"Bilateral Filter (d={d}, sColor={sigma_color}, sSpace={sigma_space})"
    
    print(f"\n{'='*50}")
    print(f"  PHÂN TÍCH: {kernel_name}")
    print(f"{'='*50}")
    print(f"  Bộ lọc song biên là bộ lọc PHI TUYẾN TÍNH.")
    print(f"  Lọc trên cả miền không gian và miền màu sắc.")
    print(f"  → Giúp làm mịn vùng đồng màu nhưng GIỮ NGUYÊN biên cạnh.")
    print(f"  sigmaColor={sigma_color}: Khoảng lọc trên miền cường độ màu sắc.")
    print(f"  sigmaSpace={sigma_space}: Khoảng lọc trên miền không gian hình học.")
    print(f"{'='*50}\n")
    
    img_filtered = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    return img_filtered, None, kernel_name
