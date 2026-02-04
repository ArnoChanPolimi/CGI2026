import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import os
import sys
import random
import numpy as np
from PIL import Image, ImageOps


# --- 1. 环境初始化与随机种子固定 ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# 路径逻辑
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

# 尝试导入日志工具
try:
    from logger_utils import get_logger

    log = get_logger()
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging

# 自动新建输出文件夹
output_dir = os.path.join(project_root, "output", "6class")
os.makedirs(output_dir, exist_ok=True)


# --- 2. 核心自定义逻辑：长边压缩填充 (Long-Side Resize & Pad) ---
class LongSidePadTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # 1. 长边缩放到目标尺寸，短边等比缩放
        res = ImageOps.contain(img, (self.size, self.size))
        # 2. 填充黑边补齐为正方形 (size x size)
        return ImageOps.pad(res, (self.size, self.size), color=0)


# --- 3. 图像处理链 (训练集含四向旋转) ---
data_transforms = {
    "train": transforms.Compose(
        [
            LongSidePadTransform(256),
            transforms.CenterCrop(224),
            transforms.RandomChoice(
                [
                    transforms.RandomRotation((0, 0)),
                    transforms.RandomRotation((90, 90)),
                    transforms.RandomRotation((180, 180)),
                    transforms.RandomRotation((270, 270)),
                ]
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            LongSidePadTransform(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

# 相对路径加载数据 (data/data_6class)
data_path = os.path.join(project_root, "data", "data_6class")
if not os.path.exists(data_path):
    log.error(f"❌ 找不到数据目录: {data_path}，请确保已运行划分脚本。")
    sys.exit()

image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_path, x), data_transforms[x])
    for x in ["train", "val"]
}
dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True)
    for x in ["train", "val"]
}

class_names = image_datasets["train"].classes
num_classes = len(class_names)
log.info(f"✅ 成功加载 {num_classes} 分类数据: {class_names}")

# --- 4. 模型构建 (MobileNet_V2) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights="DEFAULT")

# 冻结特征提取层
for param in model.parameters():
    param.requires_grad = False

# 修改分类头为 6
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# --- 5. 训练函数 ---
def train_model(num_epochs=10):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    log.info(f"🚀 开始在 {device} 上进行训练...")

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())

            log.info(
                f"Epoch {epoch}/{num_epochs-1} | {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} LR: {scheduler.get_last_lr()[0]}"
            )

    return history


# 执行训练 (10个轮次以获得更好效果)
history = train_model(num_epochs=10)

# --- 6. 生成结果产出并保存到 output/6class ---
# 1. 学习曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.title("Loss Curves")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"], label="Val Acc")
plt.title("Accuracy Curves")
plt.legend()
plt.savefig(os.path.join(output_dir, "learning_curves_6class.png"))

# 2. 最终评估与混淆矩阵
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in dataloaders["val"]:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

# 保存混淆矩阵图片
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix - Chilli Leaf 6 Classes")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix_6class.png"))

# 保存分类报告文本
report = classification_report(y_true, y_pred, target_names=class_names)
with open(os.path.join(output_dir, "classification_report_6class.txt"), "w") as f:
    f.write(report)

# 保存模型权重
torch.save(model.state_dict(), os.path.join(output_dir, "chilli_model_6class.pth"))

log.info(f"✅ 所有产出已保存至: {output_dir}")
