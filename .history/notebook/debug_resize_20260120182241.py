import os
from torchvision import transforms
from PIL import Image

# --- 准备工作 ---
img_path = r"data\original_dataset_Chilli_Leaf\Cercospora_Leaf_Spot\Cercospora Leaf Spot_002.jpg"
save_dir = "tmp"
os.makedirs(save_dir, exist_ok=True)

img = Image.open(img_path).convert("RGB")

# ---------------------------------------------------------
# 1. 你的【最原始逻辑】：无旋转，只缩放裁剪
# ---------------------------------------------------------
original_logic = transforms.Compose(
    [
        transforms.Resize(256),  # 把短边缩到 256
        transforms.CenterCrop(
            224
        ),  # 从中间抠 224。这步就是“罪魁祸首”，因为它切了长图的头尾
    ]
)

# ---------------------------------------------------------
# 2. 我的【保命逻辑】：等比例缩放 + 补齐，不切边
# ---------------------------------------------------------
safe_logic = transforms.Compose(
    [
        transforms.Resize(224),  # 把长边缩到 224，短边等比例变细
        transforms.CenterCrop(224),  # 此时 CenterCrop 发现宽度不够，自动补黑边
    ]
)

# --- 执行输出 ---
out_orig = original_logic(img)
out_safe = safe_logic(img)

out_orig.save(os.path.join(save_dir, "LOGIC_ORIGINAL.jpg"))
out_safe.save(os.path.join(save_dir, "LOGIC_SAFE_PADDING.jpg"))

print(f"原图尺寸: {img.size}")
print("-" * 50)
print("请对比 tmp 文件夹下的两张图：")
print("1. LOGIC_ORIGINAL.jpg -> 你的原始逻辑。你会发现叶子很粗大，但尖端没了！")
print("2. LOGIC_SAFE_PADDING.jpg -> 保命逻辑。你会发现整根叶子都在，左右有黑边。")
print("-" * 50)
