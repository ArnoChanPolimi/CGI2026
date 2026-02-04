import os
import shutil
import random

# --- 1. 逻辑路径设定 (全相对路径) ---
# 原始数据位置
src_root = os.path.join("data", "original_dataset_Chilli_Leaf")
# 目标生成位置
target_root = "data"

# --- 2. 映射逻辑：把 5 个文件夹归类为 2 大类 ---
mapping = {
    "Bacterial_Spot": "biotic",
    "Cercospora_Leaf_Spot": "biotic",
    "Curl_Virus": "biotic",
    "Powdery_Mildew": "biotic",
    "Nutrition_Deficiency": "abiotic",
}

# --- 3. 物理准备：创建训练集和验证集文件夹 ---
splits = ["train", "val"]
categories = ["biotic", "abiotic"]

for s in splits:
    for c in categories:
        path = os.path.join(target_root, s, c)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"已创建文件夹: {path}")

# --- 4. 核心分拣逻辑 ---
split_rate = 0.8  # 80% 训练, 20% 验证

print("\n正在开始搬运照片...")

for original_folder, target_category in mapping.items():
    folder_path = os.path.join(src_root, original_folder)

    if not os.path.exists(folder_path):
        print(f"❌ 跳过：找不到文件夹 {folder_path}")
        continue

    # 获取所有图片文件
    images = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    random.shuffle(images)  # 随机打乱，保证公平

    train_count = int(len(images) * split_rate)

    for i, img_name in enumerate(images):
        # 决定去 train 还是 val
        current_split = "train" if i < train_count else "val"

        # 构建源路径和目标路径
        src_path = os.path.join(folder_path, img_name)
        # 目标文件名加上原文件夹前缀，防止重名覆盖
        new_name = f"{original_folder}_{img_name}"
        dst_path = os.path.join(target_root, current_split, target_category, new_name)

        # 执行复制
        shutil.copy(src_path, dst_path)

print("\n✅ 处理完成！")
print(f"数据已存入: {os.path.abspath(os.path.join(target_root, 'train'))} 等文件夹中。")
