import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Import các module tự định nghĩa
from utils import get_odd_integer, get_float
from visualizer import show_images, show_all_filters, compare_filters, phan_tich_kernel
from demo import demo_zero_padding, interactive_convolution_demo
import filters

def main():
    print("="*55)
    print("  CHƯƠNG TRÌNH TÌM HIỂU KERNEL LỌC ẢNH (MODULAR)")
    print("  Môn: Xử Lý Ảnh Số")
    print("="*55)
    
    # Chọn ảnh từ máy tính
    print("Đang mở cửa sổ chọn file ảnh...")
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh đầu vào",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")]
    )
    if not file_path:
        print("Lỗi: Bạn đã hủy chọn ảnh!")
        return
    path = file_path
        
    # Đọc ảnh gốc (Hỗ trợ đường dẫn Unicode trên Windows)
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Lỗi hệ thống khi đọc ảnh: {e}")
        img = None
        
    if img is None:
        print(f"Lỗi: Không thể đọc file ảnh tại '{path}'.")
        return

    # Biến lưu kernel hiện tại cho Convolution Demo
    current_kernel = None
    current_kernel_name = ""

    while True:
        print("\n" + "="*55)
        print("  MENU LỰA CHỌN PHƯƠNG PHÁP LỌC ẢNH")
        print("  1. Bộ lọc trung bình (Average Blur)")
        print("  2. Bộ lọc Gaussian (Gaussian Blur)")
        print("  3. Bộ lọc trung vị (Median Filter)")
        print("  4. Bộ lọc song biên / 2 chiều (Bilateral Filter)")
        print("  5. Bộ lọc Làm sắc nét (Sharpen)")
        print("  6. Bộ lọc Laplacian (Phát hiện biên bậc 2)")
        print("  7. Bộ lọc Sobel (Phát hiện biên bậc 1)")
        print("  8. Tự nhập Kernel tùy chỉnh (NxN)")
        print("  9. Hiển thị tất cả bộ lọc (Tổng quát 3x3)")
        print(" 10. Minh họa phép tích chập (Convolution Demo)")
        print(" 11. So sánh 2-3 bộ lọc cạnh nhau")
        print("  0. Thoát")
        print("="*55)
        
        choice = input("Nhập lựa chọn (0-11): ").strip()
        
        if choice == '0':
            print("Đã thoát chương trình.")
            break
        elif choice == '9':
            show_all_filters(img)
            continue
        elif choice == '10':
            if current_kernel is None:
                print("\n⚠ Bạn chưa chọn bộ lọc nào có kernel!")
                print("  Hãy chọn 1 bộ lọc (1, 2, 5, 8) trước, rồi quay lại chọn 10.")
                continue
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            # Trực quan Padding
            demo_zero_padding(img_gray, current_kernel.shape[0])
            # Trực quan Tích chập tương tác
            interactive_convolution_demo(img, current_kernel, current_kernel_name)
            continue
        elif choice == '11':
            compare_filters(img)
            continue
            
        img_filtered = None
        kernel_name = ""
        kernel = None
        
        if choice == '1':
            ksize = get_odd_integer("Nhập kích thước bộ lọc (lẻ)", 3)
            img_filtered, kernel, kernel_name = filters.apply_average_blur(img, ksize)
            phan_tich_kernel(kernel, kernel_name)
            
        elif choice == '2':
            ksize = get_odd_integer("Nhập kích thước bộ lọc Gaussian (lẻ)", 3)
            sigma = get_float("Nhập độ lệch chuẩn sigma (0 để tự động)", 0.0)
            img_filtered, kernel, kernel_name = filters.apply_gaussian_blur(img, ksize, sigma)
            phan_tich_kernel(kernel, kernel_name)
            
        elif choice == '3':
            ksize = get_odd_integer("Nhập kích thước bộ lọc trung vị (lẻ >= 3)", 3)
            img_filtered, kernel, kernel_name = filters.apply_median_filter(img, ksize)
            
        elif choice == '4':
            d = int(get_float("Nhập đường kính lân cận d (ví dụ: 9)", 9))
            sigmaColor = get_float("Nhập sigmaColor (ví dụ: 75)", 75.0)
            sigmaSpace = get_float("Nhập sigmaSpace (ví dụ: 75)", 75.0)
            img_filtered, kernel, kernel_name = filters.apply_bilateral_filter(img, d, sigmaColor, sigmaSpace)
            
        elif choice == '5':
            img_filtered, kernel, kernel_name = filters.apply_sharpen_filter(img)
            phan_tich_kernel(kernel, kernel_name)
            
        elif choice == '6':
            ksize = get_odd_integer("Nhập kích thước ksize (lẻ: 1, 3, 5)", 3)
            img_filtered, kernel, kernel_name = filters.apply_laplacian_filter(img, ksize)
            if kernel is not None:
                phan_tich_kernel(kernel, kernel_name)
            
        elif choice == '7':
            ksize = get_odd_integer("Nhập kích thước Sobel (lẻ: 1, 3, 5, 7)", 3)
            print("Chọn hướng đạo hàm:")
            print("  1. Hướng ngang (Sobel X)")
            print("  2. Hướng dọc (Sobel Y)")
            print("  3. Cả hai (Sobel Magnitude) - Mặc định")
            dir_choice = input("Lựa chọn (1-3, mặc định 3): ").strip()
            
            img_filtered, kernel, kernel_name = filters.apply_sobel_filter(img, ksize, dir_choice)
            if kernel is not None:
                phan_tich_kernel(kernel, kernel_name)
                
        elif choice == '8':
            n = get_odd_integer("Nhập kích thước kernel NxN (lẻ: 3, 5, 7, 9)", 3)
            print(f"\nNhập ma trận Kernel {n}x{n} (mỗi dòng cách nhau bởi Enter, các số cách nhau bởi khoảng trắng):")
            try:
                rows = []
                for r in range(n):
                    row = list(map(float, input(f"  Dòng {r+1}: ").strip().split()))
                    if len(row) != n:
                        raise ValueError(f"Dòng {r+1} phải có đúng {n} số, nhưng nhận được {len(row)}.")
                    rows.append(row)
                raw_kernel = np.array(rows, dtype=np.float32)
                kernel_name = f"Custom Kernel {n}x{n}"
                
                he_so = input("Nhập hệ số chia chuẩn hóa (Enter để bỏ qua): ").strip()
                if he_so:
                    he_so = float(he_so)
                    if he_so != 0:
                        raw_kernel = raw_kernel / he_so
                        kernel_name += f" (chia {he_so})"
                
                img_filtered, kernel, kernel_name = filters.apply_custom_kernel(img, raw_kernel, kernel_name)
                phan_tich_kernel(kernel, kernel_name)
                    
            except Exception as e:
                print(f"Lỗi nhập liệu: {e}. Vui lòng thử lại.")
                continue
        else:
            print("Lựa chọn không hợp lệ. Vui lòng thử lại.")
            continue
        
        # Lưu kernel hiện tại để chạy Convolution Demo
        if kernel is not None:
            current_kernel = kernel
            current_kernel_name = kernel_name
            
        if img_filtered is not None:
            show_images(img, "Ảnh gốc", img_filtered, f"Kết quả: {kernel_name}",
                        kernel=kernel, kernel_name=kernel_name)

if __name__ == "__main__":
    main()
