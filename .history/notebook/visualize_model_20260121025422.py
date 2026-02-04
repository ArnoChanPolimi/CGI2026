import os
from PIL import Image, ImageOps
from torchvision import transforms
import matplotlib.pyplot as plt

# --- 1. 严格路径设置 (保持不动) ---
img_relative_path = os.path.join(
    "data", "train", "abiotic", "Nutrition_Deficiency_Nutrition Deficiency_003.jpg"
)
output_folder = "tmp"
os.makedirs(output_folder, exist_ok=True)
# 输出文件名改为更明确的对比名称
comparison_output_path = os.path.join(
    output_folder, "compression_methods_comparison.jpg"
)


# --- 2. 自定义长边对齐并填充到正方形的 Transform ---
class LongSideResizeAndPad:
    def __init__(self, size):
        self.size = size  # 目标正方形尺寸，例如 256

    def __call__(self, img):
        width, height = img.size

        # 计算长边缩放比例，使长边变为 self.size
        if width >= height:
            new_width = self.size
            new_height = int(self.size * height / width)
        else:
            new_height = self.size
            new_width = int(self.size * width / height)

        # 先进行等比例缩放
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # 然后填充黑边到 self.size x self.size 的正方形
        padded_img = ImageOps.pad(resized_img, (self.size, self.size), color=0)

        return padded_img


# --- 3. 定义两种不同的图像处理逻辑链 (都包含随机旋转) ---

# 逻辑 1: 短边对齐 -> 裁剪 (我们之前讨论并最终确定的方案)
# 优点: 保留叶片主体形状，裁剪掉周围空白
logic_short_side_align = transforms.Compose(
    [
        transforms.Resize(256),  # 短边缩到256，长边等比
        transforms.CenterCrop(224),  # 从中心切224，利用白边缓冲
        transforms.RandomChoice(
            [  # 随机旋转
                transforms.RandomRotation((0, 0)),
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((180, 180)),
                transforms.RandomRotation((270, 270)),
            ]
        ),
    ]
)

# 逻辑 2: 长边对齐 -> 填充 -> 裁剪 (你现在提出的新方案)
# 优点: 保证所有像素不丢失 (先填充)，但叶片可能会更细长
logic_long_side_align_pad = transforms.Compose(
    [
        LongSideResizeAndPad(256),  # 长边缩到256，短边等比，然后填充黑边到 256x256
        transforms.CenterCrop(224),  # 从 256x256 中切出中心 224x224 (去除填充的黑边)
        transforms.RandomChoice(
            [  # 随机旋转
                transforms.RandomRotation((0, 0)),
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((180, 180)),
                transforms.RandomRotation((270, 270)),
            ]
        ),
    ]
)

# --- 4. 执行对比并保存 ---
if not os.path.exists(img_relative_path):
    print(f"❌ 找不到文件！请检查路径：{os.path.abspath(img_relative_path)}")
else:
    original_img = Image.open(img_relative_path).convert("RGB")
    print(f"✅ 抓到图了！原始尺寸: {original_img.size}")

    # 分别生成两种处理后的图像（这里随机取一个旋转角度）
    processed_img_short_side = logic_short_side_align(original_img)
    processed_img_long_side_pad = logic_long_side_align_pad(original_img)

    # --- 5. 可视化三列对比 ---
    plt.figure(figsize=(18, 7))  # 增加图幅宽度，容纳三列

    # 列 1: 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title(
        f"1. Original Image\n{original_img.size[0]}x{original_img.size[1]}", fontsize=12
    )
    plt.axis("off")

    # 列 2: 短边对齐 -> 裁剪
    plt.subplot(1, 3, 2)
    plt.imshow(processed_img_short_side)
    plt.title(
        f"2. Short-Side Align & Crop\n{processed_img_short_side.size[0]}x{processed_img_short_side.size[1]} (Random Rotated)",
        fontsize=12,
    )
    plt.axis("off")

    # 列 3: 长边对齐 -> 填充 -> 裁剪
    plt.subplot(1, 3, 3)
    plt.imshow(processed_img_long_side_pad)
    plt.title(
        f"3. Long-Side Align & Pad & Crop\n{processed_img_long_side_pad.size[0]}x{processed_img_long_side_pad.size[1]} (Random Rotated)",
        fontsize=12,
    )
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(comparison_output_path)
    print(f"🚀 三列对比图已生成并保存至: {comparison_output_path}")

print("\n--- 任务完成 ---")
