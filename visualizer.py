import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cấu hình font tiếng Việt hiển thị đẹp và tránh lỗi ô vuông [] trong Matplotlib
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'Tahoma', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.monospace'] = ['Consolas', 'Courier New', 'DejaVu Sans Mono', 'monospace']
plt.rcParams['font.family'] = 'sans-serif'

def plot_histogram_on_ax(ax, img, title="Histogram"):
    """Vẽ histogram lên trục (ax) tương ứng"""
    if len(img.shape) == 2:
        ax.hist(img.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)
    else:
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            ax.plot(hist, color=color)
    ax.set_title(title, fontsize=9)
    ax.set_xlim([0, 256])

def draw_kernel_on_ax(ax, kernel, title="Kernel"):
    """Vẽ ma trận kernel dưới dạng heatmap lên trục matplotlib"""
    h, w = kernel.shape
    ax.imshow(kernel, cmap='RdBu_r', interpolation='nearest', aspect='equal')
    for i in range(h):
        for j in range(w):
            val = kernel[i, j]
            text_color = 'white' if abs(val) > (np.max(np.abs(kernel)) * 0.6) else 'black'
            fmt = f"{val:.2f}" if not val.is_integer() else f"{int(val)}"
            ax.text(j, i, fmt, ha='center', va='center', color=text_color,
                    fontsize=max(6, 12 - h), fontweight='bold')
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

def draw_kernel_stats_on_ax(ax, kernel, kernel_name=""):
    """Hiển thị thống kê chi tiết của kernel lên trục matplotlib"""
    ax.axis('off')
    h, w = kernel.shape
    tong = np.sum(kernel)
    center_val = kernel[h // 2, w // 2]
    
    is_symmetric = np.allclose(kernel, kernel.T)
    is_normalized = np.isclose(tong, 1.0, atol=1e-4)
    
    stats_text = (
        f"Tên: {kernel_name}\n"
        f"─────────────────\n"
        f"Kích thước: {h}x{w}\n"
        f"Tổng (Sum): {tong:.4f}\n"
        f"Min: {np.min(kernel):.4f}\n"
        f"Max: {np.max(kernel):.4f}\n"
        f"Trung tâm: {center_val:.4f}\n"
        f"Chuẩn hóa: {'Có' if is_normalized else 'Không'}\n"
        f"Đối xứng: {'Có' if is_symmetric else 'Không'}\n"
        f"─────────────────\n"
    )
    
    if np.isclose(tong, 1.0, atol=1e-4) and np.all(kernel >= 0):
        stats_text += "→ Bộ lọc làm mờ\n  (Low-pass filter)"
    elif np.isclose(tong, 1.0, atol=1e-4) and np.any(kernel < 0):
        stats_text += (f"→ Bộ lọc làm sắc nét\n"
                       f"  Trung tâm={center_val:.1f}\n"
                       f"  Xung quanh<0\n"
                       f"  => Tăng trọng số giữa\n"
                       f"  => Làm nổi cạnh")
    elif np.isclose(tong, 0.0, atol=1e-4):
        stats_text += (f"→ Bộ lọc phát hiện biên\n"
                       f"  Tổng=0 => vùng phẳng\n"
                       f"  trở thành đen,\n"
                       f"  chỉ giữ lại cạnh/biên")
    elif tong > 1.0:
        stats_text += "→ Tăng sáng tổng thể"
    else:
        stats_text += "→ Hiệu ứng đặc biệt"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', family='Segoe UI',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

def phan_tich_kernel(kernel, kernel_name=""):
    """Phân tích ý nghĩa của các hệ số trong mặt nạ lọc (in ra terminal)"""
    h, w = kernel.shape
    tong = np.sum(kernel)
    center_val = kernel[h // 2, w // 2]
    
    print(f"\n{'='*50}")
    print(f"  PHÂN TÍCH MẶT NẠ LỌC (KERNEL): {kernel_name}")
    print(f"{'='*50}")
    print(kernel)
    print(f"\n  Kích thước  : {h}x{w}")
    print(f"  Tổng (Sum)  : {tong:.4f}")
    print(f"  Min         : {np.min(kernel):.4f}")
    print(f"  Max         : {np.max(kernel):.4f}")
    print(f"  Trung tâm   : {center_val:.4f}")
    print(f"  Chuẩn hóa   : {'Có' if np.isclose(tong, 1.0, atol=1e-4) else 'Không'}")
    print(f"  Đối xứng    : {'Có' if np.allclose(kernel, kernel.T) else 'Không'}")
    print(f"{'─'*50}")
    
    if np.isclose(tong, 1.0, atol=1e-4) and np.all(kernel >= 0):
        print("  → Nhận xét: Tất cả hệ số >= 0, tổng = 1")
        print("    Đây là bộ lọc thông thấp (Low-pass) → làm mờ/khử nhiễu.")
        print("    Mỗi pixel đầu ra = trung bình có trọng số các pixel lân cận.")
    elif np.isclose(tong, 1.0, atol=1e-4) and np.any(kernel < 0):
        print(f"  → Nhận xét: Pixel trung tâm = {center_val:.1f}, xung quanh có giá trị âm")
        print(f"    Trung tâm lớn ({center_val:.1f}) → tăng trọng số điểm giữa.")
        print(f"    Xung quanh âm → trừ bớt thành phần mờ.")
        print(f"    => Kết quả: Làm nổi cạnh / sắc nét (Sharpen).")
    elif np.isclose(tong, 0.0, atol=1e-4):
        print("  → Nhận xét: Tổng hệ số = 0")
        print("    Vùng có cường độ đồng đều → output = 0 (đen).")
        print("    Chỉ vùng thay đổi cường độ đột ngột mới cho output ≠ 0.")
        print("    => Kết quả: Phát hiện biên/cạnh (Edge Detection).")
    elif tong > 1.0:
        print(f"  → Nhận xét: Tổng = {tong:.2f} > 1 → Tăng sáng tổng thể.")
    else:
        print(f"  → Nhận xét: Tổng = {tong:.4f} → Hiệu ứng đặc biệt.")
    print(f"{'='*50}\n")

def show_images(img1, title1, img2, title2, kernel=None, kernel_name=""):
    """Hiển thị ảnh gốc, ảnh đã lọc, histogram, kernel heatmap và thống kê kernel"""
    disp_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if len(img1.shape) == 3 else img1
    disp_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if len(img2.shape) == 3 else img2

    if kernel is not None:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Ảnh gốc
    ax_img1 = axes[0, 0]
    if len(disp_img1.shape) == 2:
        ax_img1.imshow(disp_img1, cmap='gray')
    else:
        ax_img1.imshow(disp_img1)
    ax_img1.set_title(title1)
    ax_img1.axis('off')
    
    # Histogram ảnh gốc
    plot_histogram_on_ax(axes[1, 0], img1, f"Histogram {title1}")

    # Ảnh sau lọc
    ax_img2 = axes[0, 1]
    if len(disp_img2.shape) == 2:
        ax_img2.imshow(disp_img2, cmap='gray')
    else:
        ax_img2.imshow(disp_img2)
    ax_img2.set_title(title2)
    ax_img2.axis('off')
    
    # Histogram ảnh sau lọc
    plot_histogram_on_ax(axes[1, 1], img2, f"Histogram {title2}")
    
    # Nếu có kernel
    if kernel is not None:
        draw_kernel_on_ax(axes[0, 2], kernel, f"Ma trận Kernel")
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def show_all_filters(img):
    """Hiển thị ảnh gốc và tất cả 8 phương pháp lọc trên cùng một lưới 3x3"""
    disp_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
    
    res_orig = disp_img
    
    res_avg = cv2.blur(img, (5, 5))
    res_avg_rgb = cv2.cvtColor(res_avg, cv2.COLOR_BGR2RGB) if len(res_avg.shape) == 3 else res_avg
    
    res_gauss = cv2.GaussianBlur(img, (5, 5), 0)
    res_gauss_rgb = cv2.cvtColor(res_gauss, cv2.COLOR_BGR2RGB) if len(res_gauss.shape) == 3 else res_gauss
    
    res_median = cv2.medianBlur(img, 5)
    res_median_rgb = cv2.cvtColor(res_median, cv2.COLOR_BGR2RGB) if len(res_median.shape) == 3 else res_median
    
    res_bilateral = cv2.bilateralFilter(img, 9, 75, 75)
    res_bilateral_rgb = cv2.cvtColor(res_bilateral, cv2.COLOR_BGR2RGB) if len(res_bilateral.shape) == 3 else res_bilateral
    
    kernel_sharpen = np.array([[ 0, -1,  0], [-1,  5, -1], [ 0, -1,  0]], dtype=np.float32)
    res_sharpen = cv2.filter2D(img, -1, kernel_sharpen)
    res_sharpen_rgb = cv2.cvtColor(res_sharpen, cv2.COLOR_BGR2RGB) if len(res_sharpen.shape) == 3 else res_sharpen
    
    res_laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    res_laplacian = cv2.convertScaleAbs(res_laplacian)
    res_lap_rgb = cv2.cvtColor(res_laplacian, cv2.COLOR_BGR2RGB) if len(res_laplacian.shape) == 3 else res_laplacian
    
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    res_sobel = cv2.convertScaleAbs(sobel_mag)
    res_sobel_rgb = cv2.cvtColor(res_sobel, cv2.COLOR_BGR2RGB) if len(res_sobel.shape) == 3 else res_sobel
    
    kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    res_emboss_cv = cv2.filter2D(img, cv2.CV_64F, kernel_emboss)
    res_emboss = cv2.convertScaleAbs(res_emboss_cv)
    res_emboss_rgb = cv2.cvtColor(res_emboss, cv2.COLOR_BGR2RGB) if len(res_emboss.shape) == 3 else res_emboss
    
    filters = [
        ("Ảnh Gốc", res_orig), ("Average Blur 5x5", res_avg_rgb),
        ("Gaussian Blur 5x5", res_gauss_rgb), ("Lọc Trung vị 5x5", res_median_rgb),
        ("Lọc 2 chiều (Bilateral)", res_bilateral_rgb), ("Làm sắc nét (Sharpen)", res_sharpen_rgb),
        ("Laplacian Edge", res_lap_rgb), ("Sobel Edge", res_sobel_rgb),
        ("Chạm nổi (Emboss)", res_emboss_rgb)
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
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

def compare_filters(img):
    """Cho người dùng chọn 2-3 bộ lọc để so sánh cạnh nhau"""
    filter_options = {
        '1': ("Average Blur 5x5", lambda i: cv2.blur(i, (5, 5))),
        '2': ("Gaussian Blur 5x5", lambda i: cv2.GaussianBlur(i, (5, 5), 0)),
        '3': ("Median Filter 5", lambda i: cv2.medianBlur(i, 5)),
        '4': ("Bilateral Filter", lambda i: cv2.bilateralFilter(i, 9, 75, 75)),
        '5': ("Sharpen 3x3", lambda i: cv2.filter2D(i, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32))),
        '6': ("Laplacian", lambda i: cv2.convertScaleAbs(cv2.Laplacian(i, cv2.CV_64F, ksize=3))),
        '7': ("Sobel Magnitude", lambda i: cv2.convertScaleAbs(cv2.magnitude(
            cv2.Sobel(i, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(i, cv2.CV_64F, 0, 1, ksize=3)))),
    }
    
    print("\nChọn các bộ lọc để so sánh (nhập số cách nhau bởi dấu cách, ví dụ: 1 2 3):")
    for k, (name, _) in filter_options.items():
        print(f"  {k}. {name}")
    
    choices_str = input("Lựa chọn (2-3 số): ").strip().split()
    choices = [c for c in choices_str if c in filter_options]
    
    if len(choices) < 2:
        print("Cần chọn ít nhất 2 bộ lọc để so sánh!")
        return
    if len(choices) > 3:
        choices = choices[:3]
        print("Chỉ giữ lại 3 bộ lọc đầu tiên.")
    
    n = len(choices)
    disp_orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
    
    fig, axes = plt.subplots(2, n + 1, figsize=(5 * (n + 1), 8))
    
    # Cột đầu tiên: Ảnh gốc + histogram
    if len(disp_orig.shape) == 2:
        axes[0, 0].imshow(disp_orig, cmap='gray')
    else:
        axes[0, 0].imshow(disp_orig)
    axes[0, 0].set_title("Ảnh Gốc", fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')
    plot_histogram_on_ax(axes[1, 0], img, "Histogram Gốc")
    
    # Các cột tiếp theo
    for idx, c in enumerate(choices):
        name, filter_func = filter_options[c]
        filtered = filter_func(img)
        disp_filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB) if len(filtered.shape) == 3 else filtered
        
        col = idx + 1
        if len(disp_filtered.shape) == 2:
            axes[0, col].imshow(disp_filtered, cmap='gray')
        else:
            axes[0, col].imshow(disp_filtered)
        axes[0, col].set_title(name, fontsize=10, fontweight='bold')
        axes[0, col].axis('off')
        plot_histogram_on_ax(axes[1, col], filtered, f"Histogram {name}")
    
    plt.suptitle("SO SÁNH CÁC BỘ LỌC", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
