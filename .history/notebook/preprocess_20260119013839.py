# notebook\preprocess.py
import cv2
import os
import numpy as np


def smart_preprocess(input_path, output_folder, target_size=(224, 224)):
    """
    自适应预处理：长条图切片，小图插值放大，普通图直接缩放。
    """
    img = cv2.imread(input_path)
    if img is None:
        return

    h, w = img.shape[:2]
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # 逻辑 A：针对长条图 (长宽比超过 1.5) -> 采用滑动窗口切片
    if max(h, w) / min(h, w) > 1.5:
        side = min(h, w)  # 以短边为正方形边长
        stride = int(side * 0.8)  # 设置20%的重叠，防止切断关键特征

        count = 0
        if w > h:  # 横向长条
            for x in range(0, w - side + 1, stride):
                tile = img[:, x : x + side]
                save_tile(tile, output_folder, f"{base_name}_tile_{count}", target_size)
                count += 1
        else:  # 纵向长条
            for y in range(0, h - side + 1, stride):
                tile = img[y : y + side, :]
                save_tile(tile, output_folder, f"{base_name}_tile_{count}", target_size)
                count += 1
        print(f"长条图处理完成: {base_name} -> 切分为 {count} 块")

    # 逻辑 B：针对普通图或 100x100 小图 -> 直接缩放
    else:
        # 使用 INTER_CUBIC 插值，这对 100x100 放大到 224 效果最好
        save_tile(img, output_folder, f"{base_name}_resized", target_size)
        print(f"常规图处理完成: {base_name}")


def save_tile(img, folder, name, size):
    resized = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(folder, f"{name}.jpg"), resized)


# 使用示例 (你可以根据你的文件夹结构修改)
raw_abiotic = "data/raw/Abiotic"
processed_abiotic = "data/processed/train/Abiotic"
os.makedirs(processed_abiotic, exist_ok=True)

for file in os.listdir(raw_abiotic):
    smart_preprocess(os.path.join(raw_abiotic, file), processed_abiotic)
