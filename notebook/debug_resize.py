import os
from torchvision import transforms
from PIL import Image

# --- 准备工作 ---
img_path = r"data\original_dataset_Chilli_Leaf\Cercospora_Leaf_Spot\Cercospora Leaf Spot_002.jpg"
save_dir = "tmp"
os.makedirs(save_dir, exist_ok=True)
img = Image.open(img_path).convert("RGB")

# ---------------------------------------------------------
# 逻辑一：【你的原始逻辑】(不带旋转，但存在“铡刀”风险)
# 功能：直接缩放裁剪。
# 结果：由于 Resize(256) 还是太大，长边有 560+，CenterCrop 必切头尾。
# ---------------------------------------------------------
original_logic = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224)]
)

# ---------------------------------------------------------
# 逻辑二：【合理旋转逻辑】(带 30 度旋转，且保全叶尖)
# ---------------------------------------------------------
safe_rotation_logic = transforms.Compose(
    [
        # [步骤 1] 先缩到 150。
        # 道理：150 的对角线是 212，小于 224。这样旋转时，尖端绝不会超出 224。
        transforms.Resize(150),
        # [步骤 2] 旋转 30 度。
        # expand=True: 旋转时自动撑大画布，防止边角丢失。
        # fill=0: 旋转后露出的三角形区域填充黑色。
        transforms.RandomRotation(degrees=30, expand=True, fill=0),
        # [步骤 3] 归位补齐。
        # 将旋转后的（可能变大到 210x210 的图）放进 224x224。
        # 因为它小于 224，所以 CenterCrop 此时的作用是：【补齐剩余的黑边】。
        transforms.CenterCrop(224),
    ]
)

# --- 执行输出 ---
# 1. 保存你的原始逻辑图
original_logic(img).save(os.path.join(save_dir, "LOGIC_ORIGINAL_CUT.jpg"))

# 2. 保存三张合理旋转后的图
for i in range(3):
    safe_rotation_logic(img).save(os.path.join(save_dir, f"LOGIC_SAFE_ROT_{i}.jpg"))

print("处理完成！请对比：")
print("1. LOGIC_ORIGINAL_CUT.jpg -> 看看叶尖是不是被切了？")
print("2. LOGIC_SAFE_ROT.jpg -> 看看虽然有黑色三角形区域，但叶子是不是 100% 完整？")
