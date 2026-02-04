# notebook/train/train_main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import sys

# 导入上面两个脚本
from dataset import PlantDataset, get_transforms
from model import get_plant_model

# 自动处理路径
current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# ================= 配置区 (针对 Windows CPU) =================
CSV_PATH = ROOT_DIR / "output" / "dataset_index" / "dataset_index.csv"
SAVE_DIR = ROOT_DIR / "output" / "train_results"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8  # CPU 建议 8，如果你内存 32G 觉得太轻松可以改 16
EPOCHS = 10  # 10 轮起步
LR = 0.0005  # 学习率小一点，学得更精细
DEVICE = torch.device("cpu")  # 强制 CPU
# ===========================================================


def train():
    print(f"--- 正在初始化训练逻辑 (设备: {DEVICE}) ---")

    # 1. 解决数据量不平衡逻辑
    df = pd.read_csv(CSV_PATH)
    train_df = df[df["split"] == "train"]
    counts = train_df["label"].value_counts().sort_index().values
    # 权重计算：总数 / 某类数量。少的类别分到的 weights 会很大。
    weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    weights = weights / weights.sum()
    class_weights = weights.to(DEVICE)
    print(f"数据量统计: Abiotic={counts[0]}, Biotic={counts[1]}")
    print(f"补偿权重已生成: {class_weights}")

    # 2. 加载数据
    train_tf, test_tf = get_transforms()
    train_ds = PlantDataset(CSV_PATH, ROOT_DIR, split="train", transform=train_tf)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    # 3. 初始化模型、损失函数和优化器
    model = get_plant_model(num_classes=2).to(DEVICE)

    # 这里的 weight=class_weights 是解决数据不平衡的关键！
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4. 开始训练循环
    print("--- 开始训练 ---")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # 前向计算
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        print(f"===> Epoch {epoch+1} 完成! 平均损失: {avg_loss:.4f}")

        # 5. 保存当前模型
        torch.save(model.state_dict(), SAVE_DIR / f"mobilenetv3_epoch_{epoch+1}.pth")

    print(f"所有训练已完成！模型保存在: {SAVE_DIR}")


if __name__ == "__main__":
    train()
