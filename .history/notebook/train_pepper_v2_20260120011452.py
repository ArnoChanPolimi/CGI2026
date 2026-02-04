import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os
import sys

# --- 1. 环境与路径逻辑 ---
# 确保能找到根目录下的 logger_utils
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from logger_utils import get_logger

log = get_logger()

# 自动创建输出文件夹（在根目录下）
output_dir = os.path.join(project_root, "output")
os.makedirs(output_dir, exist_ok=True)

# --- 2. 图像处理与数据加载 ---
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
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
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True)
    for x in ["train", "val"]
}

class_names = image_datasets["train"].classes
log.info(f"分类映射: {class_names}")

# --- 3. 模型定义 (核心：必须在调用 train_model 之前定义) ---
model = models.mobilenet_v2(weights="DEFAULT")
for param in model.parameters():
    param.requires_grad = False

# 修改分类头
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
device = torch.device("cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)

# --- 4. 记录容器 ---
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}


# --- 5. 训练函数 ---
def train_model(model, criterion, optimizer, num_epochs=5):
    log.info("开始训练...")
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

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())

            log.info(
                f"Epoch {epoch}/{num_epochs-1} | {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}"
            )


# --- 6. 执行训练 ---
train_model(model, criterion, optimizer, num_epochs=5)

# --- 7. 生成核心产出 (绘图与报告) ---
# (绘制 Loss/Acc 曲线)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"], label="Val Acc")
plt.legend()
plt.title("Accuracy Curve")
plt.savefig(os.path.join(output_dir, "learning_curves.png"))

# (评估报告)
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in dataloaders["val"]:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

report = classification_report(y_true, y_pred, target_names=class_names)
with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# 保存权重
torch.save(model.state_dict(), os.path.join(output_dir, "pepper_model.pth"))
log.info("✅ 所有核心产出已保存至 output 文件夹")
