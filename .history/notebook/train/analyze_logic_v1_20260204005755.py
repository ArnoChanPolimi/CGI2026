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
# 1. 基础目录
SAVE_DIR = ROOT_DIR / "output" / "All_crop_train_logic_v1"
VIS_OUT_DIR = SAVE_DIR / "output"  # 这里是你放图和存盘数据的地方
VIS_OUT_DIR.mkdir(parents=True, exist_ok=True)
# CSV_PATH = ROOT_DIR / "output" / "dataset_index" / "dataset_index.csv"
# CSV_PATH = ROOT_DIR / "output" / "dataset_index" / "dataset_index_letterbox_v1.csv"
# ### 修正 1：文件名必须包含 NoHealthy ###
# CSV_PATH = (
#     ROOT_DIR / "output" / "dataset_index" / "dataset_index_letterbox_NoHealthy_v1.csv"
# )
CSV_PATH = (
    ROOT_DIR
    / "output"
    / "dataset_index"
    / "dataset_index_letterbox_NoHealthy_v1_NoAloeVera_order.csv"
)

# ### 修正 2：必须指向 pth 子文件夹 ###
# MODEL_PATH = (
#     SAVE_DIR / "pth" / "checkpoint_epoch_letterbox_12.pth"
# )  # 原版 PTH12（最后一轮）
# MODEL_PATH = (
#     SAVE_DIR / "pth" / "checkpoint_epoch_letterbox_8.pth"
# )  # 测试 8 版, 防止过拟合
MODEL_PATH = (
    SAVE_DIR / "pth" / "checkpoint_epoch_order_14.pth"
)  # 原版 PTH12（最后一轮）
# TRAIN_LOG_PATH = SAVE_DIR / "logic_v1_training.log"
TRAIN_LOG_PATH = SAVE_DIR / "logic_v1_training_letterbox.log"

DEVICE = torch.device("cpu")

# --- ✨ 动态提取版本号逻辑 (更强壮的版本) ---
try:
    # 提取文件名中所有的数字，取【最后一个】数字串
    # 这样即使文件名是 model_v1_epoch_8.pth，也能精准拿到 8
    all_numbers = re.findall(r"(\d+)", MODEL_PATH.name)
    epoch_val = all_numbers[-1] if all_numbers else "unknown"
    ver_tag = f"pth{epoch_val}" if epoch_val != "unknown" else "unknown"
except Exception:
    ver_tag = "unknown"

# ========================== 3. 精准路径分配 (三路分流版) ==========================

# --- 📁 A路：错误分析目录 (存放 CSV 结构化数据) ---
ERROR_OUT_DIR = SAVE_DIR / "error_analysis" / ver_tag
ERROR_OUT_DIR.mkdir(parents=True, exist_ok=True)

ERROR_CSV_FILE = (
    ERROR_OUT_DIR / f"Logic_v1_all_misclassified_cases_{ver_tag}_letterbox.csv"
)
CROP_REPORT_FILE = (
    ERROR_OUT_DIR / f"Logic_v1_crop_error_diagnosis_{ver_tag}_letterbox.csv"
)

# --- 📁 B路：图像输出目录 (VIS_OUT_DIR 已定义为 SAVE_DIR / "output") ---
# 确保这里使用了 VIS_OUT_DIR，让混淆矩阵和 Loss 图各就各位
LOSS_ACC_PLOT_PATH = VIS_OUT_DIR / f"loss_Acc_logic_v1_letterbox_fastVersion.png"
CM_PLOT_PATH = (
    VIS_OUT_DIR / f"confusion_matrix_logic_v1_{ver_tag}_letterbox_fastVersion.png"
)

# --- 📁 C路：日志输出目录 (LOG_OUT_DIR 已定义为 ROOT_DIR / "log" / "analyze" / ver_tag) ---
# 重新获取 logger，让它把 F1 报告写到你指定的 log 文件夹里
LOG_OUT_DIR = ROOT_DIR / "log" / "analyze" / ver_tag
LOG_OUT_DIR.mkdir(parents=True, exist_ok=True)
logger = get_logger(
    log_dir=str(LOG_OUT_DIR),
    log_filename=f"logic_v1_analysis_{ver_tag}_letterbox_fastVersion.log",
)


# ========================== 2. 绘制 Loss 曲线 (科学回溯补完版) ==========================
# ========================== 2. 绘制 Loss 曲线 (带进度条与数据存盘版) ==========================
def analyze():
    logger.info(f"--- 🚀 开始 Logic V1 科学分析 (高性能存档版) ---")
    try:  # <--- 必须在这里加这一句！！
        # --- 1. 核心路径定义 --- (固定名称，不受 pth 版本影响) ---
        METRICS_CSV = (
            VIS_OUT_DIR / "backtrack_metrics_logic_v1_letterbox_fastVersion.csv"
        )

        # 尝试读取原始 Log 数据 (获取 Train Loss)
        # 2. 读取原始 Log 数据 (直接在大 try 下运行，去掉之前的小 try)
        with open(TRAIN_LOG_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        step_losses = [float(x) for x in re.findall(r"Loss: ([\d.]+)", content)]
        epoch_avg_losses = [
            float(x) for x in re.findall(r"Avg Loss: ([\d.]+)", content)
        ]

        train_acc_list, test_losses_backtrack, test_accs_backtrack = [], [], []

        # --- 2. 核心判断：如果 CSV 已存在，直接跳过计算地狱 ---
        if METRICS_CSV.exists():
            logger.info(
                f"📂 发现已存在的测算记录: {METRICS_CSV.name}，正在秒速读取数据..."
            )
            df_metrics = pd.read_csv(METRICS_CSV)

            # 严格检查数据长度是否对得上 log 里的 epoch 数
            if len(df_metrics) == len(epoch_avg_losses):
                train_acc_list = df_metrics["train_acc"].tolist()
                test_accs_backtrack = df_metrics["test_acc"].tolist()
                test_losses_backtrack = df_metrics["test_loss"].tolist()
                epoch_avg_losses = df_metrics["train_loss"].tolist()  # 确保一致性
                logger.info("✅ 4条核心曲线数据已全部从存档对齐。")
            else:
                logger.warning("⚠️ 存档数据长度不匹配，准备重新生成...")

        # --- 3. 如果没存档，才执行全量回溯 (仅需跑这一次) ---
        if not train_acc_list:
            logger.warning(
                "🧪 未发现完整测算记录，开始 1-12 轮全量考试 (请确保 GPU 可用)..."
            )

            # 自动选择设备：优先显卡，否则 CPU
            CURRENT_DEVICE = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            logger.info(f"回溯计算使用设备: {CURRENT_DEVICE}")

            tmp_model = get_plant_model(num_classes=2).to(CURRENT_DEVICE)
            _, tmp_tf = get_transforms()

            # 减小 num_workers 避免 CPU 调度卡死，加大 batch_size 提速
            back_ds = PlantDataset(CSV_PATH, ROOT_DIR, split="test", transform=tmp_tf)
            back_loader = DataLoader(
                back_ds, batch_size=256, shuffle=False, num_workers=4
            )

            train_ds_back = PlantDataset(
                CSV_PATH, ROOT_DIR, split="train", transform=tmp_tf
            )
            train_loader_back = DataLoader(
                train_ds_back, batch_size=256, shuffle=False, num_workers=4
            )

            back_criterion = torch.nn.CrossEntropyLoss()
            pbar_epoch = tqdm(range(1, len(epoch_avg_losses) + 1), desc="总体回溯进度")

            for e in pbar_epoch:
                # pth = SAVE_DIR / "pth" / f"checkpoint_epoch_letterbox_{e}.pth"
                pth = SAVE_DIR / "pth" / f"checkpoint_epoch_order_{e}.pth"
                if pth.exists():
                    tmp_model.load_state_dict(
                        torch.load(pth, map_location=CURRENT_DEVICE)
                    )
                    tmp_model.eval()

                    # 测测试集
                    t_loss, t_corr, t_total = 0, 0, 0
                    with torch.no_grad():
                        for imgs, lbls in back_loader:
                            outs = tmp_model(imgs.to(CURRENT_DEVICE))
                            t_loss += back_criterion(
                                outs, lbls.to(CURRENT_DEVICE)
                            ).item()
                            t_corr += (
                                (torch.argmax(outs, 1) == lbls.to(CURRENT_DEVICE))
                                .sum()
                                .item()
                            )
                            t_total += lbls.size(0)
                    test_losses_backtrack.append(t_loss / len(back_loader))
                    test_accs_backtrack.append(t_corr / t_total)

                    # 测训练集
                    tr_corr, tr_total = 0, 0
                    with torch.no_grad():
                        for imgs, lbls in train_loader_back:
                            outs = tmp_model(imgs.to(CURRENT_DEVICE))
                            tr_corr += (
                                (torch.argmax(outs, 1) == lbls.to(CURRENT_DEVICE))
                                .sum()
                                .item()
                            )
                            tr_total += lbls.size(0)
                    train_acc_list.append(tr_corr / tr_total)
                else:
                    test_losses_backtrack.append(None)
                    test_accs_backtrack.append(None)
                    train_acc_list.append(None)

            # 立即存档
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
            logger.info(f"💾 存档成功: {METRICS_CSV.name}")

        # --- 4. 绘图逻辑 (直接从变量读数据，极速出图) ---
        plt.figure(figsize=(15, 12))
        epochs = range(1, len(epoch_avg_losses) + 1)

        # Subplot 1: Step Loss
        plt.subplot(2, 2, 1)
        plt.plot(step_losses, color="#2ecc71", alpha=0.3, label="Batch Loss")
        if len(step_losses) > 100:
            smooth_loss = pd.Series(step_losses).rolling(window=50).mean()
            plt.plot(smooth_loss, color="#27ae60", label="Smoothed Trend")
        plt.title("Step Training Loss")
        plt.grid(True, ls="--")
        plt.legend()

        # Subplot 2: Loss Comparison
        plt.subplot(2, 2, 2)
        plt.plot(epochs, epoch_avg_losses, "r-o", label="Train Loss")
        plt.plot(epochs, test_losses_backtrack, "b-s", label="Test Loss")
        plt.title("Loss Curve: Train vs Test")
        plt.grid(True, ls="--")
        plt.legend()

        # Subplot 3: Acc Comparison
        plt.subplot(2, 1, 2)
        plt.plot(
            epochs,
            train_acc_list,
            marker="o",
            color="#f1c40f",
            linewidth=2,
            label="Train Acc",
        )
        plt.plot(
            epochs,
            test_accs_backtrack,
            marker="D",
            color="#9b59b6",
            linewidth=2,
            label="Test Acc",
        )
        for i, v in zip(epochs, test_accs_backtrack):
            if v is not None:
                plt.text(i, v + 0.01, f"{v:.2f}", ha="center", fontweight="bold")
        plt.title("Accuracy Curve: Train vs Test")
        plt.grid(True, ls="--")
        plt.legend()

        plt.tight_layout()
        plt.savefig(LOSS_ACC_PLOT_PATH)
        logger.info(f"✅ 曲线图已更新: {LOSS_ACC_PLOT_PATH.name}")

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
