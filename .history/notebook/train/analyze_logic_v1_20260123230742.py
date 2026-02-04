import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

# 确保导入逻辑版
from dataset_logic_v1 import (
    PlantDatasetLogic as PlantDataset,
    get_logic_transforms as get_transforms,
)
from model import get_plant_model

current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parent.parent.parent
sys.path.append(str(ROOT_DIR))
from notebook.tools.logger_utils import get_logger

# ========================== 1. 路径精确对齐 ==========================
SAVE_DIR = ROOT_DIR / "output" / "All_crop_train_logic_v1"
CSV_PATH = ROOT_DIR / "output" / "dataset_index" / "dataset_index.csv"
TRAIN_LOG_PATH = SAVE_DIR / "logic_v1_training.log"
MODEL_PATH = SAVE_DIR / "checkpoint_epoch_9.pth"
DEVICE = torch.device("cpu")

# 错误分析专用目录
ERROR_OUT_DIR = SAVE_DIR / "error_analysis"
ERROR_OUT_DIR.mkdir(parents=True, exist_ok=True)

# 文件路径分配
ERROR_CSV_FILE = ERROR_OUT_DIR / "all_misclassified_cases_logic_v1.csv"
CROP_REPORT_FILE = ERROR_OUT_DIR / "crop_error_diagnosis_logic_v1.csv"
LOSS_PLOT_PATH = SAVE_DIR / "loss_trend_logic_v1.png"
CM_PLOT_PATH = SAVE_DIR / "confusion_matrix_logic_v1.png"

# 日志文件存放在主目录下 (包含 F1, Precision 等)
logger = get_logger(log_dir=str(SAVE_DIR), log_filename="logic_v1_analysis.log")


def analyze():
    logger.info("--- 🔍 开始 Logic_V1 全方位评估报告 (找回旧版数值标注) ---")

    # ========================== 2. 绘制 Loss 曲线 (完整逻辑) ==========================
    try:
        with open(TRAIN_LOG_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        step_losses = [float(x) for x in re.findall(r"Loss: ([\d.]+)", content)]
        all_avg_losses = [float(x) for x in re.findall(r"平均 Loss: ([\d.]+)", content)]
        # 确保只取最新的 N 轮（根据你的 EPOCHS 决定）
        epoch_avg_losses = all_avg_losses

        plt.figure(figsize=(15, 6))

        # 左图：Step-by-step Loss (保留细节)
        plt.subplot(1, 2, 1)
        plt.plot(step_losses, color="#2ecc71", alpha=0.3, label="Batch Loss")
        if len(step_losses) > 100:
            smooth_loss = pd.Series(step_losses).rolling(window=50).mean()
            plt.plot(smooth_loss, color="#27ae60", label="Smoothed Trend")
        plt.title(f"Step-by-Step Training Loss ({len(step_losses)} Steps)")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        # 右图：Epoch Loss (✨ 找回关键点位数值标注 ✨)
        plt.subplot(1, 2, 2)
        epochs = range(1, len(epoch_avg_losses) + 1)
        plt.plot(
            epochs,
            epoch_avg_losses,
            marker="o",
            color="#e74c3c",
            linewidth=2,
            markersize=8,
        )

        # 重新加入旧版的数值标注循环
        for i, v in zip(epochs, epoch_avg_losses):
            plt.text(
                i,
                v + (max(epoch_avg_losses) * 0.02),
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#c0392b",
            )

        plt.title("Average Loss per Epoch")
        plt.xlabel("Epoch Number")
        plt.ylabel("Loss")
        plt.xticks(epochs)
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(LOSS_PLOT_PATH)
        logger.info(f"✅ 带数值标注的趋势图已保存至: {LOSS_PLOT_PATH.name}")
    except Exception as e:
        logger.error(f"绘图异常: {e}")

    # ========================== 3. 高性能推理 ==========================
    full_df = pd.read_csv(CSV_PATH)
    test_meta = full_df[full_df["split"] == "test"].reset_index(drop=True)
    crop_total_counts = test_meta["crop"].value_counts().to_dict()

    _, test_tf = get_transforms()
    test_ds = PlantDataset(CSV_PATH, ROOT_DIR, split="test", transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=8)

    model = get_plant_model(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds, all_labels, error_list = [], [], []
    label_map = {0: "Abiotic", 1: "Biotic"}

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="推理中")):
            outputs = model(images.to(DEVICE))
            preds = torch.argmax(outputs, dim=1)

            p_np, l_np = preds.numpy(), labels.numpy()
            all_preds.extend(p_np)
            all_labels.extend(l_np)

            for i in range(len(p_np)):
                if p_np[i] != l_np[i]:
                    idx = batch_idx * 64 + i
                    row = test_meta.iloc[idx]
                    error_list.append(
                        {
                            "Crop": row["crop"],
                            "Ground_Truth": label_map[l_np[i]],
                            "Prediction": label_map[p_np[i]],
                            "Path": row["rel_path"],
                        }
                    )

    # ========================== 4. 结果汇总与指标 (F1 等存入 Log) ==========================
    # 分类报告打印到日志 (主目录)
    report = classification_report(
        all_labels, all_preds, target_names=["Abiotic", "Biotic"]
    )
    logger.info(
        "\n" + "=" * 20 + " F1 / Precision / Recall 报告 " + "=" * 20 + "\n" + report
    )

    # 混淆矩阵图 (主目录)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Abiotic", "Biotic"],
        yticklabels=["Abiotic", "Biotic"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Logic V1)")
    plt.savefig(CM_PLOT_PATH)
    logger.info(f"✅ 混淆矩阵图已保存至: {CM_PLOT_PATH.name}")

    # ========================== 5. 错误分析与作物诊断 (保存至 error_analysis 目录) ==========================
    if error_list:
        # 1. 保存错题明细 CSV
        err_df = pd.DataFrame(error_list)
        err_df.to_csv(ERROR_CSV_FILE, index=False, encoding="utf-8-sig")

        # 2. 生成作物维度诊断报告
        crop_err_stats = err_df.groupby("Crop").size().reset_index(name="Error_Count")
        crop_err_stats["Total_In_Test"] = crop_err_stats["Crop"].map(crop_total_counts)
        crop_err_stats["Error_Rate (%)"] = (
            crop_err_stats["Error_Count"] / crop_err_stats["Total_In_Test"] * 100
        ).round(2)
        crop_err_stats = crop_err_stats.sort_values(
            by="Error_Rate (%)", ascending=False
        )

        crop_err_stats.to_csv(CROP_REPORT_FILE, index=False, encoding="utf-8-sig")

        logger.info(
            f"✅ 错题明细与作物诊断报告已存入: {ERROR_OUT_DIR.relative_to(ROOT_DIR)}"
        )
        logger.info("\n--- 作物偏科概览 ---\n" + crop_err_stats.to_string(index=False))

    logger.info("--- 🚀 分析任务圆满完成！ ---")


if __name__ == "__main__":
    analyze()
