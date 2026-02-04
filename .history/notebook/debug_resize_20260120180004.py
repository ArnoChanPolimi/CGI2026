import os
from torchvision import transforms
from PIL import Image

# --- 准备工作 ---
img_path = r"data\original_dataset_Chilli_Leaf\Cercospora_Leaf_Spot\Cercospora Leaf Spot_002.jpg"
save_dir = "tmp"
os.makedirs(save_dir, exist_ok=True)

# 加载原图
if not os.path.exists(img_path):
    print(f"错误：找不到图片 {img_path}")
    exit()
img = Image.open(img_path).convert("RGB")

# ---------------------------------------------------------
# 逻辑一：你原来的逻辑 (会切掉头尾)
# ---------------------------------------------------------
old_transform = transforms.Compose(
    [
        # 先把短边(958)缩到256，此时长边是564。图片还是很大。
        transforms.Resize(256),
        # 从564的高度里硬生生抠出中间的224。上下各170像素直接被扔进垃圾桶！
        transforms.CenterCrop(224),
    ]
)

# ---------------------------------------------------------
# 逻辑二：我建议的逻辑 (全貌保全)
# ---------------------------------------------------------
new_transform = transforms.Compose(
    [
        # 只给一个数224！它会把最长的那边(2113)缩到224。
        # 此时短边等比例缩成101。整根叶子都在，只是变小了。
        transforms.Resize(224),
        # 此时图是101x224。CenterCrop看到宽度不够224，
        # 它不会切，而是自动在左右两边补上黑边，凑成224x224。
        transforms.CenterCrop(224),
    ]
)

# --- 执行并保存 ---
img_old = old_transform(img)
img_new = new_transform(img)

img_old.save(os.path.join(save_dir, "OUTPUT_OLD_CROPPED.jpg"))
img_new.save(os.path.join(save_dir, "OUTPUT_NEW_COMPLETE.jpg"))

print(f"原图尺寸: {img.size}")
print(f"旧逻辑输出: {img_old.size} -> 存至 tmp/OUTPUT_OLD_CROPPED.jpg")
print(f"新逻辑输出: {img_new.size} -> 存至 tmp/OUTPUT_NEW_COMPLETE.jpg")
print("\n比较指南：")
print("1. 看 OLD：叶子是不是变得很粗，但尖端和屁股都没了？")
print("2. 看 NEW：叶子是不是变细了，但从头到尾都在，左右是黑边？")
