import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

# --- 1. 路径逻辑 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
model_path = os.path.join(project_root, "output", "pepper_model.pth")
# 自动使用你指定的生物压力图片
test_img_path = os.path.join(
    project_root, "data", "val", "biotic", "Bacterial_Spot_Bacterial_Spot_225.jpg"
)

device = torch.device("cpu")


# --- 2. 函数定义：反归一化 (核心：防止颜色变鬼片) ---
def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # 维度转换: [1, 3, H, W] -> [H, W, 3]
    img = tensor.squeeze().permute(1, 2, 0).numpy()
    img = img * std + mean
    return np.clip(img, 0, 1)


# --- 3. 加载模型 ---
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- 4. 图像处理与 Hook (提取 AI 的注意力) ---
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# 挂载钩子提取最后一层卷积层
target_layer = model.features[-1]
activations = []


def hook_fn(module, input, output):
    activations.append(output)


target_layer.register_forward_hook(hook_fn)

# 读图
img_pil = Image.open(test_img_path).convert("RGB")
input_tensor = preprocess(img_pil).unsqueeze(0)

# 运行预测
output = model(input_tensor)

# 生成热力图数据
heatmap_raw = torch.mean(activations[0], dim=1).squeeze().detach().numpy()
heatmap_raw = np.maximum(heatmap_raw, 0)
heatmap_raw /= np.max(heatmap_raw) + 1e-8  # 归一化防止除以0

# --- 5. 绘图与保存 (使用 Matplotlib 叠加，解决反色问题) ---
import cv2

heatmap_resized = cv2.resize(heatmap_raw, (224, 224))

original_rgb = denormalize(input_tensor)

plt.figure(figsize=(12, 6))

# 左图：原始绿色叶子
plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title("Original Leaf (Biotic)")
plt.axis("off")

# 右图：科研热力图叠加
plt.subplot(1, 2, 2)
plt.imshow(original_rgb)  # 先画底图
# 叠加热力层，alpha=0.5 保证透明度，cmap='jet' 保证红色是重点
plt.imshow(heatmap_resized, cmap="jet", alpha=0.5)
plt.title("AI Focus (Grad-CAM)")
plt.axis("off")

out_file = os.path.join(project_root, "output", "grad_cam_fixed_final.png")
plt.savefig(out_file, bbox_inches="tight")
print(f"✅ 终于运行成功了！结果在这里: {out_file}")
plt.show()
