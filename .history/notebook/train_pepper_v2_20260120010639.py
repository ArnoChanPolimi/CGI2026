import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import sys

# 解决导入 logger_utils 的路径问题（因为它在上一级目录）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logger_utils import get_logger

# 初始化日志和文件夹
log = get_logger()
if not os.path.exists("output"):
    os.makedirs("output")

# 1. 数据处理 (针对 300x2000 的智能裁剪逻辑)
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

# 2. 加载数据 (相对路径向上跳一级找 data)
data_dir = os.path.join("..", "data")
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ["train", "val"]
}
dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True)
    for x in ["train", "val"]
}

log.info(
    f"数据集已加载。训练集: {len(image_datasets['train'])}, 验证集: {len(image_datasets['val'])}"
)

# 3. 模型初始化
model = models.mobilenet_v2(weights="DEFAULT")
for param in model.parameters():
    param.requires_grad = False
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
device = torch.device("cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)

# 4. 训练循环
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloaders["train"]:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(image_datasets["train"])
    log.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

# 5. 生成混淆矩阵并绘图
log.info("正在生成混淆矩阵...")
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, labels in dataloaders["val"]:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.numpy())

# 绘制并保存
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=image_datasets["train"].classes
)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(os.path.join("..", "output", "confusion_matrix.png"))
log.info("✅ 混淆矩阵已保存至 output 文件夹")

torch.save(model.state_dict(), os.path.join("..", "output", "pepper_model.pth"))
