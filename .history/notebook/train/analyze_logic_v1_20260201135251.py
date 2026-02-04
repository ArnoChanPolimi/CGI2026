# notebook\train\analyze_logic_v1.py
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
# CSV_PATH = ROOT_DIR / "output" / "dataset_index" / "dataset_index.csv"
# CSV_PATH = ROOT_DIR / "output" / "dataset_index" / "dataset_index_letterbox_v1.csv"
# ### 修正 1：文件名必须包含 NoHealthy ###
CSV_PATH = (
    ROOT_DIR / "output" / "dataset_index" / "dataset_index_letterbox_NoHealthy_v1.csv"
)

# ### 修正 2：必须指向 pth 子文件夹 ###
# MODEL_PATH = SAVE_DIR / "pth" / "checkpoint_epoch_letterbox_12.pth" # 原版 PTH12（最后一轮）
MODEL_PATH = (
    SAVE_DIR / "pth" / "checkpoint_epoch_letterbox_8.pth"
)  # 测试 8 版, 防止过拟合
# TRAIN_LOG_PATH = SAVE_DIR / "logic_v1_training.log"
TRAIN_LOG_PATH = SAVE_DIR / "logic_v1_training_letterbox.log"

DEVICE = torch.device("cpu")

# --- ✨ 动态提取版本号逻辑 ---
try:
    # 自动从 "checkpoint_epoch_12.pth" 提取出 "12"
    epoch_val = re.findall(r"epoch_(\d+)", MODEL_PATH.name)[0]
    ver_tag = f"pth{epoch_val}"
except:
    ver_tag = "unknown"

# 错误分析专用目录
ERROR_OUT_DIR = SAVE_DIR / "error_analysis"
ERROR_OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- ✨ 文件路径分配 (加入 ver_tag) ---
ERROR_CSV_FILE = (
    ERROR_OUT_DIR / f"all_misclassified_cases_logic_v1_{ver_tag}_letterbox.csv"
)
CROP_REPORT_FILE = (
    ERROR_OUT_DIR / f"crop_error_diagnosis_logic_v1_{ver_tag}_letterbox.csv"
)
LOSS_PLOT_PATH = SAVE_DIR / f"loss_Acc_logic_v1_letterbox.png"
CM_PLOT_PATH = SAVE_DIR / f"confusion_matrix_logic_v1_{ver_tag}_letterbox.png"

# 日志文件名也建议对齐版本，方便回溯
logger = get_logger(
    log_dir=str(SAVE_DIR), log_filename=f"logic_v1_analysis_{ver_tag}_letterbox.log"
)


# ========================== 2. 绘制 Loss 曲线 (科学回溯补完版) ==========================
# ========================== 2. 绘制 Loss 曲线 (带进度条与数据存盘版) ==========================
def analyze():
    logger.info(f"--- 🚀 开始 Logic V1 科学分析 (带数据存档功能) ---")
    try:
        with open(TRAIN_LOG_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        step_losses = [float(x) for x in re.findall(r"Loss: ([\d.]+)", content)]
        epoch_avg_losses = [
            float(x) for x in re.findall(r"平均 Loss: ([\d.]+)", content)
        ]

        # --- ✨ 额外增加的日志存档路径 ---
        METRICS_CSV = SAVE_DIR / f"backtrack_metrics_logic_v1_{ver_tag}.csv"

        train_acc_list = []
        test_losses_backtrack = []
        test_accs_backtrack = []

        # 检查是否已经跑过测算，如果有，直接读日志
        if METRICS_CSV.exists():
            logger.info(
                f"📂 发现已存在的 Acc/Loss 记录文件: {METRICS_CSV.name}，正在直接读取..."
            )
            df_metrics = pd.read_csv(METRICS_CSV)
            train_acc_list = df_metrics["train_acc"].tolist()
            test_accs_backtrack = df_metrics["test_acc"].tolist()
            test_losses_backtrack = df_metrics["test_loss"].tolist()
        else:
            # 如果没有日志，则执行全量回溯测算
            logger.info("🧪 未发现测算记录，开始 1-12 轮全量考试 (仅需跑这一次)...")
            tmp_model = get_plant_model(num_classes=2).to(DEVICE)
            _, tmp_tf = get_transforms()

            back_ds = PlantDataset(CSV_PATH, ROOT_DIR, split="test", transform=tmp_tf)
            back_loader = DataLoader(
                back_ds, batch_size=128, shuffle=False, num_workers=12
            )

            train_ds_back = PlantDataset(
                CSV_PATH, ROOT_DIR, split="train", transform=tmp_tf
            )
            train_loader_back = DataLoader(
                train_ds_back, batch_size=128, shuffle=False, num_workers=12
            )

            back_criterion = torch.nn.CrossEntropyLoss()

            # 使用 tqdm 显示总体进度
            pbar_epoch = tqdm(range(1, len(epoch_avg_losses) + 1), desc="总体轮次进度")
            for e in pbar_epoch:
                # pth = SAVE_DIR / f"checkpoint_epoch_letterbox_{e}.pth"
                pth = SAVE_DIR / "pth" / f"checkpoint_epoch_letterbox_{e}.pth"
                if pth.exists():
                    pbar_epoch.set_description(f"正在评估第 {e}/12 轮")
                    tmp_model.load_state_dict(torch.load(pth, map_location=DEVICE))
                    tmp_model.eval()

                    # 测测试集
                    t_loss, t_corr, t_total = 0, 0, 0
                    with torch.no_grad():
                        for imgs, lbls in tqdm(
                            back_loader, desc=f"  L{e} 测试集", leave=False
                        ):
                            outs = tmp_model(imgs.to(DEVICE))
                            t_loss += back_criterion(outs, lbls.to(DEVICE)).item()
                            t_corr += (
                                (torch.argmax(outs, 1) == lbls.to(DEVICE)).sum().item()
                            )
                            t_total += lbls.size(0)
                    test_losses_backtrack.append(t_loss / len(back_loader))
                    test_accs_backtrack.append(t_corr / t_total)

                    # 测训练集 (补全缺失的日志数据)
                    tr_corr, tr_total = 0, 0
                    with torch.no_grad():
                        for imgs, lbls in tqdm(
                            train_loader_back, desc=f"  L{e} 训练集", leave=False
                        ):
                            outs = tmp_model(imgs.to(DEVICE))
                            tr_corr += (
                                (torch.argmax(outs, 1) == lbls.to(DEVICE)).sum().item()
                            )
                            tr_total += lbls.size(0)
                    train_acc_list.append(tr_corr / tr_total)
                else:
                    test_losses_backtrack.append(None)
                    test_accs_backtrack.append(None)
                    train_acc_list.append(None)

            # --- ✨ 关键步：算完立即写成 CSV 日志，防止下次重跑 ---
            df_save = pd.DataFrame(
                {
                    "epoch": range(1, len(train_acc_list) + 1),
                    "train_loss": epoch_avg_losses,
                    "test_loss": test_losses_backtrack,
                    "train_acc": train_acc_list,
                    "test_acc": test_accs_backtrack,
                }
            )
            df_save.to_csv(METRICS_CSV, index=False)
            logger.info(f"💾 测算结果已存入日志文件: {METRICS_CSV.name}")

        # --- 绘图部分 (完全保留原有风格) ---
        plt.figure(figsize=(15, 12))
        epochs = range(1, len(epoch_avg_losses) + 1)

        # 1. Step Loss
        plt.subplot(2, 2, 1)
        plt.plot(step_losses, color="#2ecc71", alpha=0.3, label="Batch Loss")
        if len(step_losses) > 100:
            smooth_loss = pd.Series(step_losses).rolling(window=50).mean()
            plt.plot(smooth_loss, color="#27ae60", label="Smoothed Trend")
        plt.title("Step Training Loss")
        plt.legend()
        plt.grid(True, ls="--")

        # 2. Loss 对比
        plt.subplot(2, 2, 2)
        plt.plot(epochs, epoch_avg_losses, "r-o", label="Train Loss")
        plt.plot(epochs, test_losses_backtrack, "b-s", label="Test Loss")
        plt.title("Loss Curve: Train vs Test")
        plt.legend()
        plt.grid(True, ls="--")

        # 3. Acc 对比 (黄紫双线)
        plt.subplot(2, 1, 2)
        plt.plot(
            epochs,
            train_acc_list,
            marker="o",
            color="#f1c40f",
            linewidth=2,
            label="Train Acc (Calculated)",
        )
        plt.plot(
            epochs,
            test_accs_backtrack,
            marker="D",
            color="#9b59b6",
            linewidth=2,
            label="Test Acc (Calculated)",
        )
        for i, v in zip(epochs, test_accs_backtrack):
            if v is not None:
                plt.text(i, v + 0.01, f"{v:.2f}", ha="center", fontweight="bold")
        plt.title("Accuracy Curve: Train vs Test")
        plt.legend()
        plt.grid(True, ls="--")

        plt.tight_layout()
        plt.savefig(LOSS_PLOT_PATH)
        logger.info(f"✅ 科学对比图已生成: {LOSS_PLOT_PATH.name}")

    except Exception as e:
        logger.error(f"绘图异常: {e}")

    # ========================== 3. 高性能推理 ==========================
    full_df = pd.read_csv(CSV_PATH)
    test_meta = full_df[full_df["split"] == "test"].reset_index(drop=True)
    crop_total_counts = test_meta["crop"].value_counts().to_dict()

    _, test_tf = get_transforms()
    test_ds = PlantDataset(CSV_PATH, ROOT_DIR, split="test", transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=12)

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
                    idx = batch_idx * 128 + i
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
    logger.info(f"========================== NEW RUN =============================")
    analyze()
