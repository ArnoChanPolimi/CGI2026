# notebook\preprocess\process_data.py
import cv2
import os
import numpy as np
from pathlib import Path

# ================= 配置区域 =================
# 原始数据路径
input_root = r"D:\AA_POLIMI\POLIMI_STUDYING\SEM3\COMMUNICATION_IN_GREEN_INFRASTRUCTURES\CGI_PROJECT\RAW_DATA"
# 处理后保存路径
output_root = r"D:\AA_POLIMI\POLIMI_STUDYING\SEM3\COMMUNICATION_IN_GREEN_INFRASTRUCTURES\CGI_PROJECT\data_processed"

# 参数设置
TARGET_SIZE = 224  # 最终输出的小图尺寸
RESIZE_SHORT = 448  # 预处理：短边缩放到448
STRIDE = 112  # 步长 (224的一半，实现50%重叠)
VARIANCE_THRESHOLD = (
    300  # 纹理方差阈值：低于此值判定为纯色背景（黑、白、灰色），直接丢弃
)
# ===========================================


def process_image(img_path, save_dir, base_name):
    # 读取图片 (支持中文路径)
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return

    h, w = img.shape[:2]

    # 1. 缩放逻辑：短边缩放到448，长边等比缩放
    if h < w:
        new_h = RESIZE_SHORT
        new_w = int(w * (RESIZE_SHORT / h))
    else:
        new_w = RESIZE_SHORT
        new_h = int(h * (RESIZE_SHORT / w))

    img_res = cv2.resize(img, (new_w, new_h))

    # 2. 滑动窗口切割
    count = 0
    for y in range(0, new_h - TARGET_SIZE + 1, STRIDE):
        for x in range(0, new_w - TARGET_SIZE + 1, STRIDE):
            patch = img_res[y : y + TARGET_SIZE, x : x + TARGET_SIZE]

            # 3. 核心过滤逻辑：计算方差
            # 如果这一块是纯黑、纯白或者纹理极少的泥土，方差会非常低
            gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            variance = np.var(gray_patch)

            if variance > VARIANCE_THRESHOLD:
                count += 1
                save_name = f"{base_name}_{count}.jpg"
                save_path = os.path.join(save_dir, save_name)
                # 保存图片 (支持中文路径)
                cv2.imencode(".jpg", patch)[1].tofile(save_path)


def start_processing():
    print("开始处理数据，请稍候...")

    # 遍历 01_Chilli, 02_Tomato 等
    for crop_folder in os.listdir(input_root):
        crop_path = os.path.join(input_root, crop_folder)
        if not os.path.isdir(crop_path):
            continue

        # 遍历 abiotic 和 biotic
        for category in ["abiotic", "biotic"]:
            category_path = os.path.join(crop_path, category)
            if not os.path.exists(category_path):
                continue

            # 创建输出目录：data_processed/01_Chilli/abiotic/
            target_dir = os.path.join(output_root, crop_folder, category)
            os.makedirs(target_dir, exist_ok=True)

            # 遍历具体的病害子目录 (如 Chilli Nutrition Deficiency)
            for sub_dir in os.listdir(category_path):
                sub_dir_path = os.path.join(category_path, sub_dir)
                if not os.path.isdir(sub_dir_path):
                    continue

                print(f"正在处理: {crop_folder} -> {category} -> {sub_dir}")

                # 遍历图片文件
                for file in os.listdir(sub_dir_path):
                    if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        img_path = os.path.join(sub_dir_path, file)
                        base_name = os.path.splitext(file)[0]
                        process_image(img_path, target_dir, base_name)

    print("✅ 处理完成！请在 data_processed 文件夹查看结果。")


if __name__ == "__main__":
    start_processing()
