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
from PIL import Image, ImageOps

# --- 1. 环境与路径逻辑 (保持不动) ---
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from logger_utils import get_logger

log = get_logger()

output_dir = os.path.join(project_root, "output")
os.makedirs(output_dir, exist_ok=True)


# --- 2. 自定义逻辑：长边缩放并填充 (Long-Side Resize & Pad) ---
class LongSidePadTransform:
    def __init__(self, size):
        self.size = size  # 目标正方形尺寸，如 256

    def __call__(self, img):
        # 步骤 A: 保持比例缩放，确保长边等于 size，短边等比缩小
        # ImageOps.contain 会处理好所有比例逻辑
        res = ImageOps.contain(img, (self.size, self.size))

        # 步骤 B: 填充黑色背景，将图片补齐为 size x size 的正方形
        # 这样能确保整片叶子都在 256x256 的画布中央
        return ImageOps.pad(res, (self.size, self.size), color=0)


# --- 3. 图像处理逻辑优化 (改为长边填充逻辑) ---
data_transforms = {
    "train": transforms.Compose(
        [
            LongSidePadTransform(256),  # 核心修改：长边缩放并填充黑边，绝不切断叶尖
            transforms.CenterCrop(224),  # 从填充好的图中切出核心 224
            transforms.RandomChoice(
                [  # 旋转逻辑保持：在正方形基础上无损旋转
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
            LongSidePadTransform(256),  # 验证集逻辑必须与训练集一致
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

data_path = os.path.join(project_root, "data")
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_path, x), data_transforms[x])
    for x in ["train", "val"]
}

dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True)
    for x in ["train", "val"]
}

class_names = image_datasets["train"].classes
log.info(f"分类映射: {class_names}")

# --- 4. 模型定义 (保持不动) ---
model = models.mobilenet_v2(weights="DEFAULT")
for param in model.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
device = torch.device("cpu")
model = model.to(device)
# --- 技能加载逻辑 ---
old_pth = os.path.join(output_dir, "pepper_model_longside.pth")
if os.path.exists(old_pth):
    try:
        # 加载旧技能
        model.load_state_dict(torch.load(old_pth, map_location=device))
        log.info(f"成功加载旧技能: {old_pth}，将在 97% 的基础上继续冲刺！")
    except:
        log.warning("旧技能结构不匹配，将开启全新训练。")

# --- 修改前的代码 ---
# criterion = nn.CrossEntropyLoss()

# --- 修改后的代码 (针对 5:1 的样本量进行加权) ---
# 注意：权重的顺序必须和你的文件夹分类顺序一致。
# 假设你的文件夹顺序是：0: abiotic, 1: biotic
weights = torch.tensor([3.0, 1.0])  # 给予少数类更高权重

# 如果你在 CPU 上跑：
device = torch.device("cpu")
criterion = nn.CrossEntropyLoss(weight=weights.to(device))

optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}


# --- 5. 训练函数 ---
def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    log.info("🚀 开始执行 [长边填充] 优化后的训练逻辑...")
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

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


# --- 6. 执行训练 ---
train_model(model, criterion, optimizer, scheduler, num_epochs=10)

# --- 7. 生成产出 (曲线、报告、混淆矩阵) ---
# 绘图逻辑完全保留
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve (Long-Side Pad)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve (Long-Side Pad)")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "learning_curves_longside.png"))

# 评估
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in dataloaders["val"]:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

# 混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, values_format="d")
plt.title("Confusion Matrix - Long-Side Pad Logic")
plt.savefig(os.path.join(output_dir, "confusion_matrix_longside.png"))

report = classification_report(y_true, y_pred, target_names=class_names)
with open(os.path.join(output_dir, "classification_report_longside.txt"), "w") as f:
    f.write(report)

torch.save(model.state_dict(), os.path.join(output_dir, "pepper_model_longside.pth"))
log.info("✅ 任务完成！长边填充逻辑已实装。")
