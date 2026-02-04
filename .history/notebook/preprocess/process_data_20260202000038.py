# notebook\preprocess\process_data.py
import cv2
import os
import numpy as np
from pathlib import Path

# ================= 相对路径配置 =================
# 获取脚本当前所在目录 (notebook/preprocess)
current_dir = Path(__file__).resolve().parent

# 自动定位根目录 (CGI_PROJECT) 并设定相对路径
# .parent.parent 代表向上跳两级
project_root = current_dir.parent.parent
input_root = project_root / "RAW_DATA"
output_root = project_root / "data_processed"

# 参数设置
TARGET_SIZE = 224
RESIZE_SHORT = 448
STRIDE = 112
VARIANCE_THRESHOLD = 300  # 过滤低纹理背景（黑、白、死色）
# ===============================================


def process_image(img_path, save_dir, base_name):
    # 使用 np.fromfile 处理可能的特殊字符路径
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return

    h, w = img.shape[:2]

    # 1. 缩放逻辑：短边缩放到448
    scale = RESIZE_SHORT / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_res = cv2.resize(img, (new_w, new_h))

    # 2. 滑动窗口切割
    count = 0
    for y in range(0, new_h - TARGET_SIZE + 1, STRIDE):
        for x in range(0, new_w - TARGET_SIZE + 1, STRIDE):
            patch = img_res[y : y + TARGET_SIZE, x : x + TARGET_SIZE]

            # 3. 纹理过滤逻辑
            gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            if np.var(gray_patch) > VARIANCE_THRESHOLD:
                count += 1
                save_name = f"{base_name}_tile_{count}.jpg"
                save_path = os.path.join(save_dir, save_name)
                # 使用 imencode 配合 tofile 保存，确保路径兼容
                cv2.imencode(".jpg", patch)[1].tofile(save_path)


def start_processing():
    if not input_root.exists():
        print(f"❌ 错误：找不到原始数据路径 {input_root}")
        return

    print(f"🚀 开始处理！\n输入路径: {input_root}\n输出路径: {output_root}")

    # 遍历 01_Chilli 等
    for crop_folder in os.listdir(input_root):
        crop_path = input_root / crop_folder
        if not crop_path.is_dir():
            continue

        for category in ["abiotic", "biotic"]:
            category_path = crop_path / category
            if not category_path.exists():
                continue

            # 创建输出：data_processed/01_Chilli/abiotic/
            target_dir = output_root / crop_folder / category
            target_dir.mkdir(parents=True, exist_ok=True)

            # 遍历子病害目录
            for sub_dir in os.listdir(category_path):
                sub_dir_path = category_path / sub_dir
                if not sub_dir_path.is_dir():
                    continue

                print(f"正在清洗: {crop_folder} -> {category} -> {sub_dir}")

                for file in os.listdir(sub_dir_path):
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        img_full_path = sub_dir_path / file
                        base_name = img_full_path.stem
                        process_image(img_full_path, str(target_dir), base_name)

    print("✅ 处理完成！你可以去 data_processed 检查切片质量了。")


if __name__ == "__main__":
    start_processing()
