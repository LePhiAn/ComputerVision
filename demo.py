import cv2
import numpy as np
import matplotlib.pyplot as plt
from visualizer import draw_kernel_on_ax

def demo_zero_padding(img_gray, kernel_size):
    """Minh họa trực quan quá trình Zero Padding trên một vùng nhỏ của ảnh"""
    pad = kernel_size // 2
    crop_size = 6
    crop = img_gray[:crop_size, :crop_size].astype(np.float64)
    
    padded = np.pad(crop, pad, mode='constant', constant_values=0)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Ảnh gốc (crop)
    ax0 = axes[0]
    ax0.imshow(crop, cmap='gray', vmin=0, vmax=255)
    for i in range(crop.shape[0]):
        for j in range(crop.shape[1]):
            ax0.text(j, i, f"{int(crop[i, j])}", ha='center', va='center',
                     color='red', fontsize=8, fontweight='bold')
    ax0.set_title(f"Ảnh gốc (vùng {crop_size}x{crop_size})", fontsize=10)
    ax0.set_xticks(range(crop.shape[1]))
    ax0.set_yticks(range(crop.shape[0]))
    ax0.grid(True, linewidth=0.5, color='yellow', alpha=0.5)
    
    # Sau zero padding
    ax1 = axes[1]
    ax1.imshow(padded, cmap='gray', vmin=0, vmax=255)
    ph, pw = padded.shape
    for i in range(ph):
        for j in range(pw):
            is_pad = (i < pad or i >= ph - pad or j < pad or j >= pw - pad)
            color = 'blue' if is_pad else 'red'
            ax1.text(j, i, f"{int(padded[i, j])}", ha='center', va='center',
                     color=color, fontsize=7, fontweight='bold')
    ax1.set_title(f"Sau Zero Padding (pad={pad}) → {ph}x{pw}", fontsize=10)
    ax1.set_xticks(range(pw))
    ax1.set_yticks(range(ph))
    ax1.grid(True, linewidth=0.5, color='yellow', alpha=0.5)
    
    # Sơ đồ minh họa
    ax2 = axes[2]
    ax2.axis('off')
    diagram = (
        f"ZERO PADDING\n"
        f"{'─'*30}\n\n"
        f"  Ảnh gốc: {crop_size}x{crop_size}\n\n"
        f"     ↓ Thêm viền 0\n"
        f"       (pad = {pad})\n\n"
        f"  Ảnh padded: {ph}x{pw}\n\n"
        f"{'─'*30}\n"
        f"Công thức:\n"
        f"  output_size = input + 2*pad\n"
        f"  {ph} = {crop_size} + 2×{pad}\n\n"
        f"Mục đích:\n"
        f"  Giữ nguyên kích thước\n"
        f"  ảnh sau khi tích chập\n"
        f"  với kernel {kernel_size}x{kernel_size}.\n\n"
        f"Màu XANH = vùng padding (=0)\n"
        f"Màu ĐỎ   = pixel gốc"
    )
    ax2.text(0.1, 0.95, diagram, transform=ax2.transAxes,
             fontsize=9, verticalalignment='top', family='Segoe UI',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle("MINH HỌA ZERO PADDING", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

def demo_convolution(img, kernel, y, x):
    """Minh họa phép tích chập tại pixel (y, x) trên ảnh xám"""
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()
    
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    padded = np.pad(img_gray.astype(np.float64), ((pad_h, pad_h), (pad_w, pad_w)),
                    mode='constant', constant_values=0)
    
    py, px = y + pad_h, x + pad_w
    region = padded[py - pad_h:py + pad_h + 1, px - pad_w:px + pad_w + 1]
    
    element_wise = region * kernel
    output_val = np.sum(element_wise)
    output_clipped = np.clip(output_val, 0, 255)
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    # 1. Vùng lân cận
    ax0 = axes[0]
    ax0.imshow(region, cmap='gray', vmin=0, vmax=255)
    for i in range(kh):
        for j in range(kw):
            ax0.text(j, i, f"{int(region[i, j])}", ha='center', va='center',
                     color='red', fontsize=max(6, 11 - kh), fontweight='bold')
    ax0.set_title(f"Vùng lân cận ({kh}x{kw})\ntại pixel ({y},{x})", fontsize=9)
    ax0.set_xticks(range(kw))
    ax0.set_yticks(range(kh))
    ax0.grid(True, linewidth=0.5, color='yellow', alpha=0.3)
    
    # 2. Kernel
    ax1 = axes[1]
    draw_kernel_on_ax(ax1, kernel, f"Kernel ({kh}x{kw})")
    
    # 3. Element-wise
    ax2 = axes[2]
    ax2.imshow(element_wise, cmap='RdBu_r', interpolation='nearest')
    for i in range(kh):
        for j in range(kw):
            val = element_wise[i, j]
            text_color = 'white' if abs(val) > (np.max(np.abs(element_wise)) * 0.5) else 'black'
            ax2.text(j, i, f"{val:.1f}", ha='center', va='center',
                     color=text_color, fontsize=max(5, 10 - kh), fontweight='bold')
    ax2.set_title("Nhân từng phần tử\n(Element-wise ×)", fontsize=9)
    ax2.set_xticks(range(kw))
    ax2.set_yticks(range(kh))
    ax2.grid(True, linewidth=0.5, color='yellow', alpha=0.3)
    
    # 4. Kết quả
    ax3 = axes[3]
    ax3.axis('off')
    result_text = (
        f"KẾT QUẢ TÍCH CHẬP\n"
        f"{'─'*28}\n\n"
        f"Pixel gốc tại ({y},{x}):\n"
        f"  Giá trị = {int(img_gray[y, x])}\n\n"
        f"Phép tính:\n"
        f"  Σ (vùng × kernel)\n"
        f"  = {output_val:.2f}\n\n"
        f"Sau clip [0, 255]:\n"
        f"  = {int(output_clipped)}\n\n"
        f"{'─'*28}\n"
        f"Pixel gốc:  {int(img_gray[y, x])}\n"
        f"Pixel mới:  {int(output_clipped)}\n"
        f"Thay đổi:   {int(output_clipped) - int(img_gray[y, x]):+d}"
    )
    ax3.text(0.1, 0.95, result_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', family='Segoe UI',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.9))
    
    plt.suptitle(f"MINH HỌA PHÉP TÍCH CHẬP TẠI PIXEL ({y}, {x})", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

def interactive_convolution_demo(img, kernel, kernel_name):
    """Cho người dùng click chuột chọn pixel trên ảnh, rồi minh họa tích chập"""
    disp_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    if len(disp_img.shape) == 2:
        ax.imshow(disp_img, cmap='gray')
    else:
        ax.imshow(disp_img)
    ax.set_title("CLICK CHUỘT vào 1 điểm ảnh để xem minh họa tích chập\n(Nhấn chuột trái 1 lần)", fontsize=12)
    ax.axis('on')
    
    print("\n>>> Hãy CLICK CHUỘT vào 1 điểm trên ảnh hiển thị...")
    print(">>> (Sau khi click, đóng cửa sổ ảnh để xem kết quả tích chập)\n")
    
    try:
        coords = plt.ginput(1, timeout=60)
        plt.close(fig)
        
        if coords:
            px, py = coords[0]
            x, y = int(round(px)), int(round(py))
            h_img, w_img = img.shape[:2]
            kh = kernel.shape[0] // 2
            x = max(kh, min(x, w_img - 1 - kh))
            y = max(kh, min(y, h_img - 1 - kh))
            
            print(f">>> Đã chọn pixel ({y}, {x}). Đang minh họa tích chập...")
            demo_convolution(img, kernel, y, x)
        else:
            print(">>> Không có điểm nào được chọn (hết thời gian chờ).")
    except Exception as e:
        plt.close(fig)
        print(f"Lỗi: {e}")
