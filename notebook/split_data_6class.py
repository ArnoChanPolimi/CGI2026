import os
import shutil
import random

# --- 1. 定死随机种子，确保你每次运行划分的结果都一模一样 ---
random.seed(42)

# --- 2. 纯相对路径设定 ---
src_root = os.path.join("data", "original_dataset_Chilli_Leaf")
target_root = os.path.join("data", "data_6class")

# 获取 6 个类别的文件夹名
if not os.path.exists(src_root):
    print(f"❌ 找不到源目录: {src_root}")
else:
    categories = [
        f for f in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, f))
    ]

    # --- 3. 划分并分发 ---
    for category in categories:
        folder_path = os.path.join(src_root, category)

        # 简单过滤：只拿照片，排除杂质文件
        images = [
            f
            for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        random.shuffle(images)

        # 80/20 划分
        train_count = int(len(images) * 0.8)

        for i, img_name in enumerate(images):
            split = "train" if i < train_count else "val"
            dst_dir = os.path.join(target_root, split, category)

            # 自动新建：没有这个文件夹就立刻建一个，不管嵌套多少层
            os.makedirs(dst_dir, exist_ok=True)

            # 纯粹搬运：不改名，不改路径逻辑
            shutil.copy(
                os.path.join(folder_path, img_name), os.path.join(dst_dir, img_name)
            )

    print(f"✅ 划分完成！Seed已设为42，数据存放在: {target_root}")
