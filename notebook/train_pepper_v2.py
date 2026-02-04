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

# --- 1. 环境与路径逻辑 (保持不动) ---
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from logger_utils import get_logger

log = get_logger()

output_dir = os.path.join(project_root, "output")
os.makedirs(output_dir, exist_ok=True)

# --- 2. 图像处理逻辑优化 (吸取逻辑教训：保全全貌 + 四向无损旋转) ---
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize(256),  # 保护长宽比，利用白边缓冲
            transforms.CenterCrop(224),  # 物理锁死叶子主体，裁掉多余空白
            # 核心逻辑：针对海量异构数据，加入无像素损失的四向随机旋转
            transforms.RandomChoice(
                [
                    transforms.RandomRotation((0, 0)),
                    transforms.RandomRotation((90, 90)),
                    transforms.RandomRotation((180, 180)),
                    transforms.RandomRotation((270, 270)),
                ]
            ),
            transforms.RandomHorizontalFlip(),  # 进一步增加样本多样性
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
# 针对几万张数据，batch_size 可以适当根据显存调整，这里保持 8 (CPU)
dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True)
    for x in ["train", "val"]
}

class_names = image_datasets["train"].classes
log.info(f"分类映射: {class_names}")

# --- 3. 模型定义 ---
model = models.mobilenet_v2(weights="DEFAULT")
for param in model.parameters():
    param.requires_grad = False

# 修改分类头
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
device = torch.device("cpu")  # 如果有GPU，建议改为 "cuda"
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)

# --- 新增：学习率调度器 (防止后期 Acc 停滞) ---
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# --- 4. 记录容器 (保持不动) ---
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}


# --- 5. 训练函数 ---
def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    log.info("开始执行优化后的训练逻辑...")
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

            # 在训练阶段结束后调整学习率
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
train_model(model, criterion, optimizer, scheduler, num_epochs=5)

# --- 7. 生成核心产出 (绘图逻辑保持不动) ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "learning_curves_shortside.png"))
log.info("📈 学习曲线已保存至 output/learning_curves_shortside.png")

# 评估与混淆矩阵
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in dataloaders["val"]:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, values_format="d")
plt.title("Confusion Matrix - Chilli Leaf Classification(short side pad logic)")
plt.savefig(os.path.join(output_dir, "confusion_matrix_shortside.png"))
log.info("📊 混淆矩阵图已保存至 output/confusion_matrix_shortside.png")

# 保存文本报告与模型
report = classification_report(y_true, y_pred, target_names=class_names)
with open(os.path.join(output_dir, "classification_report_shortside.txt"), "w") as f:
    f.write(report)

torch.save(model.state_dict(), os.path.join(output_dir, "pepper_model_shortside.pth"))
log.info("✅ short side 任务全部完成！逻辑已按照要求最终定型。")
