import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import cv2

# --- 1. 路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
model_path = os.path.join(project_root, "output", "pepper_model.pth")

# 你想测哪张图，就改下面这一行
test_img_path = os.path.join(
    project_root,
    "data",
    "val",
    "biotic",
    "Nutrition_Deficiency_Nutrition Deficiency_009.png",
)

device = torch.device("cpu")

# --- 2. 分类名称 (必须和训练时的文件夹顺序对应) ---
# 0: abiotic, 1: biotic
class_names = ["Abiotic Stress (Non-biological)", "Biotic Stress (Biological)"]


# --- 3. 函数定义 ---
def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.squeeze().permute(1, 2, 0).numpy()
    img = img * std + mean
    return np.clip(img, 0, 1)


# --- 4. 加载模型 ---
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- 5. 挂载钩子 (提取 AI 注意力) ---
target_layer = model.features[-1]
activations = []


def hook_fn(module, input, output):
    activations.append(output)


target_layer.register_forward_hook(hook_fn)

# --- 6. 图片预处理与推理 ---
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

img_pil = Image.open(test_img_path).convert("RGB")
input_tensor = preprocess(img_pil).unsqueeze(0)

# AI 开始判断
output = model(input_tensor)
_, predicted = torch.max(output, 1)
pred_idx = predicted.item()  # 拿到分类结果索引

# 生成热力图
heatmap_raw = torch.mean(activations[0], dim=1).squeeze().detach().numpy()
heatmap_raw = np.maximum(heatmap_raw, 0)
heatmap_raw /= np.max(heatmap_raw) + 1e-8

# --- 7. 绘图与结论显示 ---
heatmap_resized = cv2.resize(heatmap_raw, (224, 224))
original_rgb = denormalize(input_tensor)

plt.figure(figsize=(12, 7))

# 左图：原始图
plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title(f"Input: {os.path.basename(test_img_path)}")
plt.axis("off")

# 右图：结论 + 热力图
plt.subplot(1, 2, 2)
plt.imshow(original_rgb)
plt.imshow(heatmap_resized, cmap="jet", alpha=0.4)  # 叠加半透明热力图
plt.title(f"AI PREDICTION: {class_names[pred_idx]}\n(Red area = Evidence)")
plt.axis("off")

# 保存结论图
out_file = os.path.join(project_root, "output", "final_inference_result.png")
plt.savefig(out_file, bbox_inches="tight")

print("-" * 30)
print(f"【判定结果】: {class_names[pred_idx]}")
print(f"【证据图片】: 已保存至 {out_file}")
print("-" * 30)

plt.show()
