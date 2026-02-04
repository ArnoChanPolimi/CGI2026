# notebook\train\analyze_results.py
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

# 导入你的模块
from dataset import PlantDataset, get_transforms
from model import get_plant_model

# 处理路径和导入你的 logger
current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parent.parent.parent
sys.path.append(str(ROOT_DIR))
from notebook.tools.logger_utils import get_logger

# 1. 路径配置
SAVE_DIR = ROOT_DIR / "output" / "All_crop_train"
CSV_PATH = ROOT_DIR / "output" / "dataset_index" / "dataset_index.csv"
TRAIN_LOG_PATH = SAVE_DIR / "cpu_training_v1.log"
# MODEL_PATH = SAVE_DIR / "checkpoint_epoch_5.pth"
# 修改为：
MODEL_PATH = SAVE_DIR / "checkpoint_epoch_7.pth"
DEVICE = torch.device("cpu")

# --- 初始化你的专属 Logger ---
# 文件名设为 output_analys.log
logger = get_logger(log_dir=str(SAVE_DIR), log_filename="output_analys.log")


def analyze():
    logger.info("--- 🔍 开始全方位结果分析报告 ---")

    # ========================== 1. 绘制 Loss 曲线 ==========================
    logger.info("正在从训练日志中提取数据并绘图...")
    try:
        with open(TRAIN_LOG_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        # 提取 Batch Loss (所有的 Step)
        step_losses = [float(x) for x in re.findall(r"Loss: ([\d.]+)", content)]

        # 提取每轮平均 Loss (兼容新旧格式)
        all_avg_losses = [float(x) for x in re.findall(r"平均 Loss: ([\d.]+)", content)]

        # 【去重】：如果你中间断点重跑过，日志会有重叠。
        # 这里我们假设你最终完成了 7 轮，取最后 7 条记录是最准的。
        epoch_avg_losses = (
            all_avg_losses[-7:] if len(all_avg_losses) >= 7 else all_avg_losses
        )

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        # 如果 Step 太多导致图太乱，可以只取最后一部分
        plt.plot(step_losses, color="#2ecc71", alpha=0.3, label="Batch Loss")
        plt.title(f"Training Loss ({len(step_losses)} Steps)")
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.subplot(1, 2, 2)
        plt.plot(
            range(1, len(epoch_avg_losses) + 1),
            epoch_avg_losses,
            marker="o",
            linestyle="-",
            color="#e74c3c",
            linewidth=2,
            label="Avg Loss",
        )
        # 在点上标注具体数值，方便看最后两轮降了多少
        for i, v in enumerate(epoch_avg_losses):
            plt.text(i + 1, v, f"{v:.4f}", ha="center", va="bottom")

        plt.title("Average Loss per Epoch (1-7)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xticks(range(1, len(epoch_avg_losses) + 1))
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(SAVE_DIR / "loss_trend_v2.png")
        logger.info(f"✅ 包含 1-7 轮的趋势图已保存至: loss_trend_v2.png")
    except Exception as e:
        logger.error(f"绘图失败: {e}")

    # ========================== 2. 运行测试集验证 (性能压榨版) ==========================
    logger.info(f"正在加载模型: {MODEL_PATH.name}")
    _, test_tf = get_transforms()
    test_ds = PlantDataset(CSV_PATH, ROOT_DIR, split="test", transform=test_tf)

    # 压榨性能：batch_size=64, num_workers=8
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False, num_workers=8, pin_memory=True
    )

    model = get_plant_model(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    logger.info("正在扫描测试集图片 (8线程并行)...")
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ========================== 3. 生成指标并记录日志 ==========================
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=["Abiotic", "Biotic"]
    )

    # 使用 logger 记录核心指标 (这会自动存入 output_analys.log)
    logger.info("\n" + "=" * 20 + " 分类报告 " + "=" * 20 + "\n" + report)

    # 记录混淆矩阵原始数据
    matrix_msg = (
        f"\n混淆矩阵原始数据:\n"
        f"预测 ->\tAbiotic\tBiotic\n"
        f"实际Abiotic\t{cm[0][0]}\t{cm[0][1]}\n"
        f"实际Biotic\t{cm[1][0]}\t{cm[1][1]}"
    )
    logger.info(matrix_msg)

    # 绘制混淆矩阵图
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Abiotic", "Biotic"],
        yticklabels=["Abiotic", "Biotic"],
    )
    plt.title("Confusion Matrix")
    plt.savefig(SAVE_DIR / "confusion_matrix.png")

    logger.info("✅ 混淆矩阵图已保存。")
    logger.info("--- 分析任务圆满完成！ ---")
    plt.show()


if __name__ == "__main__":
    analyze()
