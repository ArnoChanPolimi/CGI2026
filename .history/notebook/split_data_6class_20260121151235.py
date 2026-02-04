import os
import shutil
import random

# --- 1. 路径设定 ---
# 原始数据位置
# 原始数据位置
src_root = os.path.join("data", "original_dataset_Chilli_Leaf")
# 目标位置：在 data 文件夹下新建 data_6class
target_root = os.path.join("data", "data_6class")

# --- 2. 获取 6 个原始文件夹名字 ---
categories = [
    f for f in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, f))
]

# --- 3. 创建文件夹结构 ---
splits = ["train", "val"]
for s in splits:
    for c in categories:
        path = os.path.join(target_root, s, c)
        if not os.path.exists(path):
            os.makedirs(path)

# --- 4. 核心分拣逻辑 (80/20 比例，不改文件名) ---
split_rate = 0.8

print(f"🚀 开始分拣六分类数据到: {target_root}")

for category in categories:
    folder_path = os.path.join(src_root, category)

    # 获取图片列表
    images = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    random.shuffle(images)

    train_count = int(len(images) * split_rate)

    for i, img_name in enumerate(images):
        # 决定去 train 还是 val
        current_split = "train" if i < train_count else "val"

        src_path = os.path.join(folder_path, img_name)
        dst_path = os.path.join(target_root, current_split, category, img_name)

        # 执行复制 (不修改文件名)
        shutil.copy(src_path, dst_path)

print(f"\n✅ 处理完成！6类数据已存入: {os.path.abspath(target_root)}")
