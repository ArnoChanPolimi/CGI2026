import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import cv2

# --- 1. 路径逻辑（严格匹配你的训练脚本） ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
# 模型在 output 文件夹里
model_path = os.path.join(project_root, "output", "pepper_model.pth")
# 随便找一张验证集里的辣椒图做测试
test_img_path = os.path.join(
    project_root, "data", "val", "biotic", "Bacterial_Spot_Bacterial_Spot_008.png"
)

device = torch.device("cpu")  # 保持一致用 CPU

# --- 2. 加载模型（结构必须和训练时一模一样） ---
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- 3. 预处理 ---
data_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# --- 4. Grad-CAM 核心逻辑（中间过程的可视化） ---
# 解释：我们要看 MobileNetV2 的 features 里的最后一层卷积层
target_layer = model.features[-1]

# 钩子：负责把 AI 的中间思维“偷”出来
activations = []


def hook_fn(module, input, output):
    activations.append(output)


target_layer.register_forward_hook(hook_fn)

# 读取并处理图片
img_pil = Image.open(test_img_path).convert("RGB")
input_tensor = data_transform(img_pil).unsqueeze(0)

# 运行模型
output = model(input_tensor)

# 提取特征图并生成热力图
heatmap_raw = torch.mean(activations[0], dim=1).squeeze().detach().numpy()
heatmap_raw = np.maximum(heatmap_raw, 0)
heatmap_raw /= np.max(heatmap_raw)

# --- 5. 修正后的绘图逻辑 (不再胡扯，直接出图) ---

# 1. 还原底图颜色 (确保是正常的绿色叶子)
original_rgb = denormalize(input_tensor)

# 2. 开始画图
plt.figure(figsize=(12, 6))

# 左侧：原始图
plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title("Original Pepper Leaf (Biotic)")
plt.axis("off")

# 右侧：热力图叠加 (这就是你要的科研图)
plt.subplot(1, 2, 2)
plt.imshow(original_rgb)  # 先铺一层正常的底图

# 直接在底图上盖一层热力图，alpha=0.5 表示半透明，cmap='jet' 保证红的是重点，蓝的是忽略
plt.imshow(heatmap_resized, cmap="jet", alpha=0.5)
plt.title("AI Focus Area (Grad-CAM)")
plt.axis("off")

# 保存结果
final_out = os.path.join(project_root, "output", "final_check.png")
plt.savefig(final_out, bbox_inches="tight")
print(f"✅ 终于搞定了！去看这张图: {final_out}")
plt.show()
