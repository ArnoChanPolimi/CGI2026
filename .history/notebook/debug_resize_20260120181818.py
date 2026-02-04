import os
from torchvision import transforms
from PIL import Image

# --- 1. 准备图片 ---
img_path = r"data\original_dataset_Chilli_Leaf\Cercospora_Leaf_Spot\Cercospora Leaf Spot_002.jpg"
save_dir = "tmp"
os.makedirs(save_dir, exist_ok=True)

if not os.path.exists(img_path):
    print(f"死活找不到图: {img_path}")
    exit()
img = Image.open(img_path).convert("RGB")

# --- 2. 逻辑一：【旧逻辑】(Resize 256 + 旋转 + CenterCrop) ---
# 这个逻辑会把你的长叶子剪成秃头！
old_transform = transforms.Compose(
    [
        transforms.Resize(256),  # 缩放不够，长边依然很长
        transforms.RandomRotation(45),  # 旋转时角会超出边框
        transforms.CenterCrop(224),  # 硬生生切掉超出 224 的部分
    ]
)

# --- 3. 逻辑二：【新逻辑】(Resize 150 + 旋转expand + CenterCrop) ---
# 这个逻辑给旋转预留了巨大空间，保证叶尖绝对安全！
new_transform = transforms.Compose(
    [
        transforms.Resize(150),  # 关键点：缩到足够小(150)，给旋转留出对角线空间
        transforms.RandomRotation(45, expand=True),  # 关键点：expand 撑大画布保住尖尖
        transforms.CenterCrop(
            224
        ),  # 关键点：把撑大的画布放回224，此时只会补黑边，绝不切叶子
    ]
)

# --- 4. 批量生成并对比 ---
print("正在生成对比图，请稍后...")
for i in range(3):
    # 旧逻辑输出
    old_img = old_transform(img)
    old_img.save(os.path.join(save_dir, f"COMPARE_OLD_CUT_{i}.jpg"))

    # 新逻辑输出
    new_img = new_transform(img)
    new_img.save(os.path.join(save_dir, f"COMPARE_NEW_SAFE_{i}.jpg"))

print(f"\n对比图已生成到 {os.path.abspath(save_dir)}")
print("-" * 50)
print("【看图指南】:")
print(
    f"1. 打开 COMPARE_OLD_CUT_{i}.jpg：你会看到叶子很大，但尖端被斜着切断了，甚至直接飞出了画面。"
)
print(
    f"2. 打开 COMPARE_NEW_SAFE_{i}.jpg：你会看到整根叶子完整地斜躺在黑框里，四周全是黑边，没伤到一丁点叶肉。"
)
print("-" * 50)
