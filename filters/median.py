import cv2

def apply_median_filter(img, ksize):
    """Áp dụng bộ lọc trung vị (Median Filter)"""
    if ksize < 3:
        ksize = 3
    kernel_name = f"Median Filter {ksize}x{ksize}"
    
    print(f"\n{'='*50}")
    print(f"  PHÂN TÍCH: {kernel_name}")
    print(f"{'='*50}")
    print(f"  Bộ lọc trung vị là bộ lọc PHI TUYẾN TÍNH.")
    print(f"  Không sử dụng kernel ma trận số như các bộ lọc tuyến tính.")
    print(f"  Hoạt động: Sắp xếp {ksize*ksize} pixel trong cửa sổ {ksize}x{ksize},")
    print(f"  chọn giá trị ở vị trí chính giữa (trung vị).")
    print(f"  → Khử cực tốt nhiễu muối tiêu (salt-and-pepper).")
    print(f"{'='*50}\n")
    
    img_filtered = cv2.medianBlur(img, ksize)
    return img_filtered, None, kernel_name
