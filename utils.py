def get_odd_integer(prompt, default_val):
    """Yêu cầu nhập một số nguyên dương lẻ từ terminal"""
    val_str = input(f"{prompt} (Mặc định {default_val}): ").strip()
    if not val_str:
        return default_val
    try:
        val = int(val_str)
        if val <= 0 or val % 2 == 0:
            print(f"Phải là số nguyên dương lẻ! Đã chọn mặc định: {default_val}")
            return default_val
        return val
    except ValueError:
        print(f"Nhập sai định dạng! Đã chọn mặc định: {default_val}")
        return default_val

def get_float(prompt, default_val):
    """Yêu cầu nhập một số thực từ terminal"""
    val_str = input(f"{prompt} (Mặc định {default_val}): ").strip()
    if not val_str:
        return default_val
    try:
        return float(val_str)
    except ValueError:
        print(f"Nhập sai định dạng! Đã chọn mặc định: {default_val}")
        return default_val
