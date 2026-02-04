import os
import shutil
import random

# --- 1. 路径设定 (全相对路径，严禁 D:\...) ---
# 确保脚本在项目根目录运行
src_root = os.path.join("data", "original_dataset_Chilli_Leaf")
target_root = os.path.join("data", "data_6class")

# --- 2. 检查源目录是否存在 ---
if not os.path.exists(src_root):
    print(f"❌ 错误：找不到源数据目录 {src_root}")
    # 这里可以根据你的实际环境微调，但坚持使用相对路径
else:
    # 获取 6 个原始文件夹名字
    categories = [
        f for f in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, f))
    ]

    # --- 3. 自动新建目标文件夹结构 ---
    splits = ["train", "val"]
    for s in splits:
        for c in categories:
            path = os.path.join(target_root, s, c)
            # 这里的 os.makedirs(..., exist_ok=True) 确保了：没有就新建，有了也不报错
            os.makedirs(path, exist_ok=True)

    # --- 4. 核心分拣逻辑 (80/20, 不改名) ---
    split_rate = 0.8
    print(f"🚀 开始分拣六分类数据到: {target_root}")

    for category in categories:
        folder_path = os.path.join(src_root, category)

        images = [
            f
            for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        random.shuffle(images)

        train_count = int(len(images) * split_rate)

        for i, img_name in enumerate(images):
            current_split = "train" if i < train_count else "val"

            src_path = os.path.join(folder_path, img_name)
            dst_path = os.path.join(target_root, current_split, category, img_name)

            # 执行复制 (严格保持原始文件名)
            shutil.copy(src_path, dst_path)

    print(f"\n✅ 处理完成！6类数据已存入: {target_root}")
