import os
from torchvision import transforms
from PIL import Image

# --- 准备工作 ---
img_path = r"data\original_dataset_Chilli_Leaf\Cercospora_Leaf_Spot\Cercospora Leaf Spot_002.jpg"
save_dir = "tmp"
os.makedirs(save_dir, exist_ok=True)

img = Image.open(img_path).convert("RGB")

# ---------------------------------------------------------
# 逻辑一：旧逻辑 + 随机旋转 (依然会切肉)
# ---------------------------------------------------------
old_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomRotation(30),  # 随机转正负30度
        transforms.CenterCrop(224),  # 依然在切肉，旋转后切得更乱
    ]
)

# ---------------------------------------------------------
# 逻辑二：新逻辑 + 随机旋转 (保命 + 灵活性)
# ---------------------------------------------------------
new_transform = transforms.Compose(
    [
        # 1. 先把长边缩到 224，短边变细。此时叶子完整。
        transforms.Resize(224),
        # 2. 随机旋转。由于现在图是长方形（比如 101x224），
        # expand=True 会确保旋转后的边角不被切掉，但图的大小会变。
        transforms.RandomRotation(30, expand=True),
        # 3. 最后的保底补齐。
        # 此时不管转成什么样，CenterCrop 都会把图放进 224x224，
        # 宽度不够就补黑边，高度够了就居中。
        transforms.CenterCrop(224),
        # 4. 加上你之前提过的水平翻转
        transforms.RandomHorizontalFlip(),
    ]
)

# --- 执行并保存 ---
# 我们循环存 3 张，让你看看随机旋转的效果
for i in range(3):
    img_old = old_transform(img)
    img_new = new_transform(img)

    img_old.save(os.path.join(save_dir, f"OLD_rotated_{i}.jpg"))
    img_new.save(os.path.join(save_dir, f"NEW_rotated_{i}.jpg"))

print(f"处理完成！请去 tmp 查看 0, 1, 2 号对比图。")
print("注意看 NEW_rotated：叶子无论怎么歪，尖端是不是都在黑框里？")
