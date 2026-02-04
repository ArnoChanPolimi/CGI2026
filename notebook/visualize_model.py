import os
from PIL import Image, ImageOps
from torchvision import transforms
import matplotlib.pyplot as plt
import random

# --- 1. 路径设置 ---
img_relative_path = os.path.join(
    "data", "train", "abiotic", "Nutrition_Deficiency_Nutrition Deficiency_003.jpg"
)
output_folder = "tmp"
os.makedirs(output_folder, exist_ok=True)
comparison_output_path = os.path.join(
    output_folder, "final_rotation_logic_comparison.jpg"
)


# --- 2. 自定义长边驱动填充逻辑 ---
class LongSidePadTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # 等比例缩放，确保长边为 size
        res = ImageOps.contain(img, (self.size, self.size))
        # 居中填充黑色到 size x size 正方形
        return ImageOps.pad(res, (self.size, self.size), color=0)


# --- 3. 定义旋转角度 (固定在 90 度倍数) ---
# 为了让对比更有意义，我们手动选一个角度（比如 90 或 180），确保两边转得一样
chosen_angle = random.choice([0, 90, 180, 270])
rot_logic = transforms.RandomRotation((chosen_angle, chosen_angle))

# 方案 A: 短边驱动逻辑
short_side_logic = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        rot_logic,  # 在正方形基础上旋转
    ]
)

# 方案 B: 长边驱动逻辑
long_side_logic = transforms.Compose(
    [
        LongSidePadTransform(256),
        transforms.CenterCrop(224),
        rot_logic,  # 在正方形基础上旋转
    ]
)

# --- 4. 执行并绘图 ---
if not os.path.exists(img_relative_path):
    print(f"❌ 找不到文件：{img_relative_path}")
else:
    orig = Image.open(img_relative_path).convert("RGB")

    img_a = short_side_logic(orig)
    img_b = long_side_logic(orig)

    plt.figure(figsize=(18, 6))

    # 原图
    plt.subplot(1, 3, 1)
    plt.imshow(orig)
    plt.title(f"1. Original\n{orig.size}")
    plt.axis("off")

    # 短边方案
    plt.subplot(1, 3, 2)
    plt.imshow(img_a)
    plt.title(f"2. Short-Side Align\n(Rotated {chosen_angle}°)")
    plt.axis("off")

    # 长边方案
    plt.subplot(1, 3, 3)
    plt.imshow(img_b)
    plt.title(f"3. Long-Side Pad\n(Rotated {chosen_angle}°)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(comparison_output_path)
    print(f"🚀 三列对比图已生成：{comparison_output_path}，旋转角度：{chosen_angle}°")
