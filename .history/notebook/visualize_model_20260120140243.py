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

# --- 5. 绘图并保存 ---
# 将热力图拉伸到原图大小
heatmap_resized = cv2.resize(heatmap_raw, (224, 224))
heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)


# 把归一化的图转回人眼能看的图
def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.squeeze().permute(1, 2, 0).numpy()
    img = img * std + mean
    return np.clip(img, 0, 1)


original_view = denormalize(input_tensor)
# 叠加
overlay = cv2.addWeighted(np.uint8(255 * original_view), 0.6, heatmap_color, 0.4, 0)


# 关键修正：将 OpenCV 的 BGR 格式转回 RGB 格式
overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_view)  # 这里显示的是正常的 RGB
plt.title("Original Leaf")

plt.subplot(1, 2, 2)
plt.imshow(overlay_rgb)  # 使用修正后的 RGB 图像
plt.title("AI Attention (Grad-CAM)")

# 保存并展示
plt.savefig(os.path.join(project_root, "output", "grad_cam_fixed.png"))
plt.show()
