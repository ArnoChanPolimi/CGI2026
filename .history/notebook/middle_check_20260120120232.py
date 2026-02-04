import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms, datasets
import os

# 1. 这里的路径逻辑必须和你训练脚本一致
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
data_path = os.path.join(project_root, "data", "train")

# 2. 定义训练时的物理逻辑（完全照搬你的代码）
train_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # 注意：Normalize 会改变颜色，为了人眼能看清，我们在后面会进行“反归一化”
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# 3. 加载一张原始图作为对比逻辑
dataset = datasets.ImageFolder(
    os.path.join(project_root, "data", "train"), transform=train_transforms
)
# 获取第一张图的原始路径（用于读取原貌）
img_path, _ = dataset.samples[0]
from PIL import Image

original_img = Image.open(img_path)

# 4. 获取经过 Transform 后的训练图
train_img_tensor, label = dataset[0]


# --- 物理逻辑：反归一化 (Denormalization) ---
# 因为训练图被减去了均值、除以了方差，直接看是黑乎乎或色彩怪异的。
# 我们必须把这个数学操作倒过来，才能看到 AI “眼中的彩色世界”。
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    res = tensor * std + mean
    return res.clamp(0, 1)  # 确保像素值在 0-1 之间


processed_img = denormalize(train_img_tensor).permute(1, 2, 0).numpy()

# 5. 绘图对比
plt.figure(figsize=(12, 6))

# 左侧：原始图（不管它是 4000x3000 还是 300x300）
plt.subplot(1, 2, 1)
plt.imshow(original_img)
plt.title(f"Original Image\nSize: {original_img.size}")
plt.axis("off")

# 右侧：最终训练图（严格的 224x224）
plt.subplot(1, 2, 2)
plt.imshow(processed_img)
plt.title("Final Training Image\n(224x224, Normalized)")
plt.axis("off")

# 保存结果
output_path = os.path.join(project_root, "output", "data_flow_check.png")
plt.savefig(output_path)
print(f"✅ 逻辑对比图已保存至: {output_path}")
plt.show()


# 直接画出归一化后的 Tensor，不还原颜色
plt.imshow(train_img_tensor.permute(1, 2, 0).numpy())
plt.title("What the Model REALLY Sees (Raw Normalized Data)")
plt.show()
