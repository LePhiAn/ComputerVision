import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def plot_histogram_on_ax(ax, img, title="Histogram"):
    """Vẽ histogram lên trục (ax) tương ứng"""
    if len(img.shape) == 2:
        # Ảnh xám
        ax.hist(img.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)
    else:
        # Ảnh màu
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            ax.plot(hist, color=color)
    ax.set_title(title)
    ax.set_xlim([0, 256])

def show_images(img1, title1, img2, title2):
    """Hiển thị 2 ảnh (ảnh gốc và ảnh đã lọc) cùng histogram cạnh nhau"""
    # Chuyển đổi BGR sang RGB nếu là ảnh màu (chỉ dùng để hiển thị, không đổi dữ liệu gốc)
    disp_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if len(img1.shape) == 3 else img1
    disp_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if len(img2.shape) == 3 else img2

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Hiển thị ảnh gốc
    if len(disp_img1.shape) == 2:
        axes[0, 0].imshow(disp_img1, cmap='gray')
    else:
        axes[0, 0].imshow(disp_img1)
    axes[0, 0].set_title(title1)
    axes[0, 0].axis('off')
    
    # Histogram ảnh gốc
    plot_histogram_on_ax(axes[1, 0], img1, f"Histogram {title1}")

    # Hiển thị ảnh sau biến đổi
    if len(disp_img2.shape) == 2:
        axes[0, 1].imshow(disp_img2, cmap='gray')
    else:
        axes[0, 1].imshow(disp_img2)
    axes[0, 1].set_title(title2)
    axes[0, 1].axis('off')
    
    # Histogram ảnh sau biến đổi
    plot_histogram_on_ax(axes[1, 1], img2, f"Histogram {title2}")
    
    plt.tight_layout()
    plt.show()

def show_all_filters(img):
    """Hiển thị ảnh gốc và tất cả các bộ lọc chính trên cùng một khung hình"""
    disp_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
    
    # 1. Ảnh gốc
    res_orig = disp_img
    
    # 2. Average Blur 5x5
    res_avg = cv2.blur(img, (5, 5))
    res_avg_rgb = cv2.cvtColor(res_avg, cv2.COLOR_BGR2RGB) if len(res_avg.shape) == 3 else res_avg
    
    # 3. Gaussian Blur 5x5
    res_gauss = cv2.GaussianBlur(img, (5, 5), 0)
    res_gauss_rgb = cv2.cvtColor(res_gauss, cv2.COLOR_BGR2RGB) if len(res_gauss.shape) == 3 else res_gauss
    
    # 4. Lọc trung vị 5x5 (Median Filter)
    res_median = cv2.medianBlur(img, 5)
    res_median_rgb = cv2.cvtColor(res_median, cv2.COLOR_BGR2RGB) if len(res_median.shape) == 3 else res_median
    
    # 5. Lọc 2 chiều (Bilateral Filter)
    res_bilateral = cv2.bilateralFilter(img, 9, 75, 75)
    res_bilateral_rgb = cv2.cvtColor(res_bilateral, cv2.COLOR_BGR2RGB) if len(res_bilateral.shape) == 3 else res_bilateral
    
    # 6. Laplacian (ksize = 3)
    res_laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    res_laplacian = cv2.convertScaleAbs(res_laplacian)
    res_lap_rgb = cv2.cvtColor(res_laplacian, cv2.COLOR_BGR2RGB) if len(res_laplacian.shape) == 3 else res_laplacian
    
    filters = [
        ("Ảnh Gốc", res_orig),
        ("Average Blur 5x5", res_avg_rgb),
        ("Gaussian Blur 5x5", res_gauss_rgb),
        ("Lọc Trung vị 5x5 (Median)", res_median_rgb),
        ("Lọc 2 chiều (Bilateral)", res_bilateral_rgb),
        ("Laplacian Edge Detection", res_lap_rgb)
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, res) in enumerate(filters):
        if len(res.shape) == 2:
            axes[i].imshow(res, cmap='gray')
        else:
            axes[i].imshow(res)
        axes[i].set_title(name)
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()

def phan_tich_kernel(kernel):
    """Phân tích ý nghĩa của các hệ số trong mặt nạ lọc"""
    print("\n--- PHÂN TÍCH MẶT NẠ LỌC (KERNEL) ---")
    print(kernel)
    tong = np.sum(kernel)
    print(f"Tổng các hệ số: {tong:.4f}")
    
    if np.isclose(tong, 1.0, atol=1e-4):
        print("-> Nhận xét: Tổng các hệ số bằng 1. Bộ lọc này giữ nguyên độ sáng trung bình của ảnh.")
        print("   Thường thấy ở các bộ lọc làm mờ (Blur, Smoothing) hoặc làm sắc nét (Sharpening) bảo toàn nền.")
    elif np.isclose(tong, 0.0, atol=1e-4):
        print("-> Nhận xét: Tổng các hệ số bằng 0. Bộ lọc này sẽ loại bỏ các vùng có độ sáng đồng đều (trở thành màu đen),")
        print("   và chỉ giữ lại sự thay đổi độ sáng (các cạnh/biên).")
        print("   Thường thấy ở các bộ lọc phát hiện biên (Edge Detection, Laplacian, Sobel).")
    elif tong > 1.0:
        print("-> Nhận xét: Tổng các hệ số lớn hơn 1. Bộ lọc này sẽ làm tăng độ sáng tổng thể của ảnh.")
    else:
        print("-> Nhận xét: Tổng các hệ số nhỏ hơn 1 hoặc âm. Bộ lọc này làm giảm độ sáng tổng thể của ảnh hoặc tạo hiệu ứng đặc biệt.")
        
    center_val = kernel[kernel.shape[0]//2, kernel.shape[1]//2]
    if center_val > 0 and np.isclose(tong, 1.0, atol=1e-4) and np.any(kernel < 0):
        print("-> Nhận xét thêm: Hệ số trung tâm dương lớn, các hệ số xung quanh âm. Đây là đặc trưng của bộ lọc làm sắc nét (Sharpen).")
    elif center_val > 0 and np.isclose(tong, 0.0, atol=1e-4) and np.any(kernel < 0):
        print("-> Nhận xét thêm: Đây là đặc trưng của bộ lọc đạo hàm (như Laplacian), dùng để tìm biên hoặc chi tiết cạnh.")
    elif np.all(kernel >= 0):
        print("-> Nhận xét thêm: Tất cả hệ số đều không âm. Đây là bộ lọc thông thấp (Low-pass filter), có tác dụng làm mờ hoặc khử nhiễu.")
    print("--------------------------------------\n")

def test_tieu_chuan(image_name):
    """Tìm ảnh trong thư mục img hoặc ngoài thư mục gốc"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(script_dir, 'img', image_name),
        os.path.join(script_dir, image_name),
        os.path.join('img', image_name),
        image_name
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def get_odd_integer(prompt, default_val):
    """Yêu cầu nhập một số nguyên dương lẻ, nếu không hợp lệ hoặc trống sẽ dùng giá trị mặc định"""
    val_str = input(f"{prompt} (Mặc định {default_val}): ").strip()
    if not val_str:
        return default_val
    try:
        val = int(val_str)
        if val <= 0 or val % 2 == 0:
            print(f"Kích thước bộ lọc phải là số nguyên dương lẻ! Đã chọn mặc định: {default_val}")
            return default_val
        return val
    except ValueError:
        print(f"Nhập sai định dạng! Đã chọn mặc định: {default_val}")
        return default_val

def get_float(prompt, default_val):
    """Yêu cầu nhập một số thực, nếu không hợp lệ hoặc trống sẽ dùng giá trị mặc định"""
    val_str = input(f"{prompt} (Mặc định {default_val}): ").strip()
    if not val_str:
        return default_val
    try:
        return float(val_str)
    except ValueError:
        print(f"Nhập sai định dạng! Đã chọn mặc định: {default_val}")
        return default_val

def main():
    print("="*50)
    print("CHÀO MỪNG ĐẾN VỚI CHƯƠNG TRÌNH TÌM HIỂU KERNEL LỌC ẢNH")
    print("="*50)
    
    # Chọn ảnh từ máy tính
    print("Đang mở cửa sổ chọn file ảnh...")
    root = tk.Tk()
    root.withdraw() # Ẩn cửa sổ gốc
    root.attributes('-topmost', True) # Đưa cửa sổ chọn file lên trên cùng
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh đầu vào",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")]
    )
    if not file_path:
        print("Lỗi: Bạn đã hủy chọn ảnh!")
        return
    path = file_path
        
    # Đọc ảnh gốc (Hỗ trợ đường dẫn có dấu Tiếng Việt/Unicode trên Windows)
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Lỗi hệ thống khi đọc ảnh: {e}")
        img = None
        
    if img is None:
        print(f"Lỗi: Không thể đọc hoặc giải mã file ảnh tại '{path}'.")
        print("Vui lòng kiểm tra tính toàn vẹn của tệp tin hoặc thử đường dẫn khác không có ký tự đặc biệt.")
        return

    while True:
        print("\n" + "="*50)
        print("MENU LỰA CHỌN PHƯƠNG PHÁP LỌC ẢNH")
        print("1. Bộ lọc trung bình (Average Blur)")
        print("2. Bộ lọc Gaussian (Gaussian Blur)")
        print("3. Bộ lọc trung vị (Median Filter) - [Mới]")
        print("4. Bộ lọc song biên / 2 chiều (Bilateral Filter) - [Mới]")
        print("5. Bộ lọc Làm sắc nét (Sharpen) 3x3")
        print("6. Bộ lọc Laplacian (Phát hiện biên bậc 2) - [Mới]")
        print("7. Bộ lọc Sobel (Phát hiện biên bậc 1) - [Mới]")
        print("8. Tự nhập Kernel 2D tùy chỉnh (Ma trận 3x3)")
        print("9. Hiển thị tất cả bộ lọc trên cùng 1 ảnh (Tổng quát)")
        print("0. Thoát")
        print("="*50)
        
        choice = input("Nhập lựa chọn của bạn (0-9): ").strip()
        
        if choice == '0':
            print("Đã thoát chương trình.")
            break
        elif choice == '9':
            show_all_filters(img)
            continue
            
        img_filtered = None
        kernel_name = ""
        
        if choice == '1':
            ksize = get_odd_integer("Nhập kích thước bộ lọc (lẻ)", 3)
            kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
            kernel_name = f"Average Blur {ksize}x{ksize}"
            phan_tich_kernel(kernel)
            img_filtered = cv2.filter2D(img, -1, kernel)
            
        elif choice == '2':
            ksize = get_odd_integer("Nhập kích thước bộ lọc Gaussian (lẻ)", 3)
            sigma = get_float("Nhập độ lệch chuẩn sigma (0 để tự động)", 0.0)
            # Tính toán ma trận kernel Gauss thực tế để hiển thị
            kernel_1d = cv2.getGaussianKernel(ksize, sigma)
            kernel = np.outer(kernel_1d, kernel_1d)
            kernel_name = f"Gaussian Blur {ksize}x{ksize} (sigma={sigma if sigma > 0 else 'Auto'})"
            phan_tich_kernel(kernel)
            img_filtered = cv2.GaussianBlur(img, (ksize, ksize), sigma)
            
        elif choice == '3':
            ksize = get_odd_integer("Nhập kích thước bộ lọc trung vị (lẻ >= 3)", 3)
            if ksize < 3:
                ksize = 3
            print("\n--- PHÂN TÍCH BỘ LỌC TRUNG VỊ (MEDIAN FILTER) ---")
            print(f"Kích thước cửa sổ lọc: {ksize}x{ksize}")
            print("Bộ lọc trung vị là bộ lọc phi tuyến tính (non-linear filter).")
            print("Nó hoạt động bằng cách sắp xếp các giá trị pixel trong cửa sổ lọc")
            print("và chọn giá trị ở vị trí chính giữa (trung vị).")
            print("-> Tác dụng: Khử cực tốt nhiễu muối tiêu (salt-and-pepper) mà không làm mờ biên.")
            print("-------------------------------------------------\n")
            img_filtered = cv2.medianBlur(img, ksize)
            kernel_name = f"Median Filter {ksize}x{ksize}"
            
        elif choice == '4':
            d = int(get_float("Nhập đường kính lân cận d (ví dụ: 9, <=0 để tự động từ sigmaSpace)", 9))
            sigmaColor = get_float("Nhập sigmaColor (miền màu sắc, ví dụ: 75)", 75.0)
            sigmaSpace = get_float("Nhập sigmaSpace (miền không gian, ví dụ: 75)", 75.0)
            print("\n--- PHÂN TÍCH BỘ LỌC SONG BIÊN / 2 CHIỀU (BILATERAL FILTER) ---")
            print(f"Tham số: d={d}, sigmaColor={sigmaColor}, sigmaSpace={sigmaSpace}")
            print("Bộ lọc song biên lọc ảnh trên cả 2 miền (không gian và cường độ màu sắc).")
            print("Nó giúp làm mịn các vùng đồng màu (giảm nhiễu) nhưng giữ lại biên cạnh sắc nét")
            print("vì khi chênh lệch cường độ màu lớn, trọng số lọc sẽ giảm về 0.")
            print("--------------------------------------------------------------\n")
            img_filtered = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
            kernel_name = f"Bilateral Filter (d={d}, sColor={sigmaColor}, sSpace={sigmaSpace})"
            
        elif choice == '5':
            kernel = np.array([[ 0, -1,  0],
                               [-1,  5, -1],
                               [ 0, -1,  0]], dtype=np.float32)
            kernel_name = "Sharpen 3x3"
            phan_tich_kernel(kernel)
            img_filtered = cv2.filter2D(img, -1, kernel)
            
        elif choice == '6':
            ksize = get_odd_integer("Nhập kích thước nhân chập ksize (lẻ, ví dụ: 1, 3, 5)", 3)
            print("\n--- PHÂN TÍCH BỘ LỌC LAPLACIAN ---")
            print(f"Kích thước ksize = {ksize}")
            print("Toán tử Laplacian tính đạo hàm bậc hai của ảnh để phát hiện biên cạnh.")
            print("Nó nhạy với mọi hướng (không phân biệt hướng biên).")
            print("----------------------------------\n")
            res_lap = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
            img_filtered = cv2.convertScaleAbs(res_lap)
            kernel_name = f"Laplacian (ksize={ksize})"
            
        elif choice == '7':
            ksize = get_odd_integer("Nhập kích thước nhân chập Sobel (lẻ: 1, 3, 5, 7)", 3)
            print("Chọn hướng đạo hàm:")
            print("1. Hướng ngang (Sobel X)")
            print("2. Hướng dọc (Sobel Y)")
            print("3. Cả hai hướng (Sobel Magnitude) - Mặc định")
            dir_choice = input("Lựa chọn (1-3, mặc định 3): ").strip()
            
            print("\n--- PHÂN TÍCH BỘ LỌC SOBEL ---")
            print(f"Kích thước ksize = {ksize}")
            print("Toán tử Sobel tính đạo hàm bậc một để xác định biên cạnh theo hướng.")
            print("-------------------------------\n")
            
            if dir_choice == '1':
                sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
                img_filtered = cv2.convertScaleAbs(sobelx)
                kernel_name = f"Sobel X (ksize={ksize})"
            elif dir_choice == '2':
                sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
                img_filtered = cv2.convertScaleAbs(sobely)
                kernel_name = f"Sobel Y (ksize={ksize})"
            else:
                sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
                sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
                sobel_magnitude = cv2.magnitude(sobelx, sobely)
                img_filtered = cv2.convertScaleAbs(sobel_magnitude)
                kernel_name = f"Sobel Magnitude (ksize={ksize})"
                
        elif choice == '8':
            print("\nNhập ma trận Kernel 3x3 (mỗi dòng cách nhau bởi Enter, các số cách nhau bởi khoảng trắng):")
            try:
                row1 = list(map(float, input("Dòng 1 (VD: 0 -1 0): ").strip().split()))
                row2 = list(map(float, input("Dòng 2 (VD: -1 5 -1): ").strip().split()))
                row3 = list(map(float, input("Dòng 3 (VD: 0 -1 0): ").strip().split()))
                if len(row1) != 3 or len(row2) != 3 or len(row3) != 3:
                    raise ValueError("Mỗi dòng phải có đúng 3 số.")
                kernel = np.array([row1, row2, row3], dtype=np.float32)
                kernel_name = "Custom Kernel 3x3"
                
                # Hỏi xem có muốn chia cho hệ số chuẩn hóa không
                he_so = input("Nhập hệ số chia (Nhấn Enter để bỏ qua hoặc chia cho 1): ").strip()
                if he_so:
                    he_so = float(he_so)
                    if he_so != 0:
                        kernel = kernel / he_so
                        kernel_name += f" (đã chia {he_so})"
                
                phan_tich_kernel(kernel)
                
                tong = np.sum(kernel)
                # Chống tràn màu cho các bộ lọc có hệ số âm (như Edge Detection, Sharpen)
                if tong <= 0 or np.any(kernel < 0):
                    res_cv = cv2.filter2D(img, cv2.CV_64F, kernel)
                    img_filtered = cv2.convertScaleAbs(res_cv)
                else:
                    img_filtered = cv2.filter2D(img, -1, kernel)
                    
            except Exception as e:
                print(f"Lỗi nhập liệu: {e}. Vui lòng thử lại.")
                continue
        else:
            print("Lựa chọn không hợp lệ. Vui lòng thử lại.")
            continue
            
        if img_filtered is not None:
            # Hiển thị
            show_images(img, "Ảnh gốc", img_filtered, f"Kết quả: {kernel_name}")

if __name__ == "__main__":
    main()
