import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

# 1. 基础配置（确保路径正确）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "pepper_model.pth"  # 指向你训练好的模型文件
img_path = "data/train/biotic/test_leaf.jpg"  # 随便找一张你想看的辣椒图路径

# 2. 加载模型结构并注入权重
# 注意：模型结构必须和训练时完全一致
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 2)  # 假设你是2类分类
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

# 3. 图像预处理（严格对应论文里的归一化）
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

img_pil = Image.open(img_path).convert("RGB")
input_tensor = preprocess(img_pil).unsqueeze(0).to(device)


# --- 功能：提取中间特征图 (Feature Map) ---
def plot_middle_features(model, x):
    # 我们看 MobileNetV2 的前几个特征层
    features = model.features[:4](x)
    features = features.detach().cpu().numpy()

    plt.figure(figsize=(10, 5))
    for i in range(8):  # 展示前8个通道
        plt.subplot(2, 4, i + 1)
        plt.imshow(features[0, i, :, :], cmap="viridis")  # 这里就是“奇怪颜色”的来源
        plt.axis("off")
    plt.suptitle("Intermediate Feature Maps (What AI sees internally)")
    plt.show()


# --- 功能：生成 Grad-CAM (注意力热力图) ---
# 这里简化了逻辑，直接提取最后一层特征的响应
def plot_grad_cam(model, x, original_img):
    # 找到 MobileNetV2 的最后一个卷积层
    target_layer = model.features[-1]

    # 钩子函数获取特征
    features = []

    def hook(module, input, output):
        features.append(output)

    handle = target_layer.register_forward_hook(hook)
    output = model(x)
    handle.remove()

    # 计算热力图
    heatmap = torch.mean(features[0], dim=1).squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)  # 归一化

    # 调整大小回 224x224
    import cv2

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 叠加到原图
    original_img_np = np.array(original_img.resize((224, 224)))
    overlayed = cv2.addWeighted(original_img_np, 0.6, heatmap, 0.4, 0)

    plt.imshow(overlayed)
    plt.title("Grad-CAM: AI Attention Area")
    plt.axis("off")
    plt.show()


# 执行可视化
plot_middle_features(model, input_tensor)
plot_grad_cam(model, input_tensor, img_pil)
