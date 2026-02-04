# notebook\test\cross_species_benchmarking_pth12.py
import torch
import pandas as pd
import numpy as np
import re
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report  # 在脚本顶部添加
from torch.utils.data import DataLoader

# ========================== 1. 动态路径与环境配置 ==========================
current_file = Path(__file__).resolve()
# 脚本在 notebook\test\，向上退两级到达项目根目录
ROOT_DIR = current_file.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# from notebook.train.dataset_logic_v1 import get_logic_transforms
# 修改为：
from notebook.train.dataset_letterbox import (
    get_letterbox_transforms as get_logic_transforms,
)
from notebook.train.model import get_plant_model
from notebook.tools.logger_utils import get_logger

# 相对路径配置
# MODEL_PATH = ROOT_DIR / "output" / "All_crop_train_logic_v1" / "checkpoint_epoch_12.pth" #
# MODEL_PATH = (
#     ROOT_DIR
#     / "output"
#     / "All_crop_train_logic_v1"
#     / "checkpoint_epoch_letterbox_12.pth"
# )  # 优化版的 PTH12 版本
# --- ✨ 核心模型路径 (修改这里，下面全部自动关联) ---
MODEL_PATH = (
    ROOT_DIR
    / "output"
    / "All_crop_train_logic_v1"
    / "pth"
    / "checkpoint_epoch_letterbox_8.pth"
)

# --- ✨ 动态提取版本标签 (如 pth8, pth12) ---
try:
    all_numbers = re.findall(r"(\d+)", MODEL_PATH.name)
    epoch_val = all_numbers[-1] if all_numbers else "unknown"
    ver_tag = f"pth{epoch_val}"
except Exception:
    ver_tag = "unknown"

# --- ✨ 重新定义输出分流路径 ---
OTHER_DATA_DIR = ROOT_DIR / "other"

# 路径 A: 结果输出目录 (按版本号分文件夹)
# 对应路径: output\All_crop_train_logic_v1\cross_species_test\Nohealthy_test\pth8
RESULT_DIR = (
    ROOT_DIR
    / "output"
    / "All_crop_train_logic_v1"
    / "cross_species_test"
    / "Nohealthy_test"
    / ver_tag
)

# 路径 B: 日志目录
# 对应路径: log\cross_species_test
LOG_DIR = ROOT_DIR / "log" / "cross_species_test"

# 创建目录
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- ✨ 日志命名自动对齐版本 ---
logger = get_logger(
    log_dir=str(LOG_DIR),
    log_filename=f"cross_species_benchmarking_{ver_tag}_NoHealthy.log",
)

DEVICE = torch.device("cpu")
LABEL_MAP = {0: "Abiotic", 1: "Biotic"}

# ========================== 2. 加载模型与配置 ==========================
logger.info("=" * 50)
logger.info(f"🚀 开始跨物种实战测试 (PTH12 版本)")
logger.info(f"模型路径: {MODEL_PATH}")
logger.info(f"数据目录: {OTHER_DATA_DIR}")
logger.info("=" * 50)

model = get_plant_model(num_classes=2).to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    logger.info("✅ 模型权重加载成功")
except Exception as e:
    logger.error(f"❌ 模型加载失败: {e}")
    sys.exit()

_, test_tf = get_logic_transforms()


def save_confusion_matrix(y_true, y_pred, species_name):
    """为指定数据集生成混淆矩阵图"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Abiotic", "Biotic"],
        yticklabels=["Abiotic", "Biotic"],
    )
    plt.title(f"Confusion Matrix: {species_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    # 文件名区分单物种和总计
    suffix = "TOTAL" if species_name == "OVERALL_ALL_SPECIES" else species_name
    save_path = RESULT_DIR / f"cm_{suffix}_pth12_letterbox.png"

    plt.savefig(save_path)
    plt.close()
    logger.info(f"📊 混淆矩阵已保存至: {save_path.name}")


# ========================== 3. 执行评估逻辑 ==========================
def run_benchmark():
    all_summary = []

    # ✨ 新增：用于存储所有物种的累积预测结果，生成总混淆矩阵
    overall_y_true = []
    overall_y_pred = []

    # 获取 other 下的所有物种文件夹
    species_folders = sorted([f for f in OTHER_DATA_DIR.iterdir() if f.is_dir()])

    for species_folder in species_folders:
        species_name = species_folder.name
        logger.info(f"\n🌿 正在测试物种: {species_name}")

        y_true, y_pred, details = [], [], []

        # 遍历目标子目录
        for target_val, sub_dir in [(0, "abiotic"), (1, "biotic")]:
            target_path = species_folder / sub_dir
            if not target_path.exists():
                logger.warning(f"⚠️ 跳过缺失目录: {target_path}")
                continue

            # 递归搜索图片
            img_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG"]:
                img_files.extend(list(target_path.rglob(ext)))

            if not img_files:
                continue

            for img_p in tqdm(img_files, desc=f"  Evaluating {sub_dir}", leave=False):
                try:
                    img = Image.open(img_p).convert("RGB")
                    input_tensor = test_tf(img).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        output = model(input_tensor)
                        pred = torch.argmax(output, 1).item()
                        prob = torch.softmax(output, 1)[0][pred].item()

                    y_true.append(target_val)
                    y_pred.append(pred)
                    details.append(
                        {
                            "FileName": img_p.name,
                            "Actual": LABEL_MAP[target_val],
                            "Predicted": LABEL_MAP[pred],
                            "Confidence": f"{prob:.4f}",
                            "SubCategory": img_p.parent.name,  # 记录具体的病害名
                        }
                    )
                except Exception as e:
                    logger.error(f"❌ 处理图片出错 {img_p.name}: {e}")

        # 统计单物种结果
        if y_true:
            acc = accuracy_score(y_true, y_pred)
            # 生成详细的分类报告（包含 Precision, Recall, F1）
            report = classification_report(
                y_true, y_pred, target_names=["Abiotic", "Biotic"], output_dict=True
            )

            f1_biotic = report["Biotic"]["f1-score"]

            logger.info(f"📊 {species_name} 汇总:")
            logger.info(f"   - Accuracy: {acc:.2%}")
            logger.info(f"   - Biotic F1: {f1_biotic:.4f}")
            logger.info(f"   - Precision: {report['Biotic']['precision']:.4f}")
            logger.info(f"   - Recall: {report['Biotic']['recall']:.4f}")

            # ✨ 将单物种结果加入全局集合
            overall_y_true.extend(y_true)
            overall_y_pred.extend(y_pred)

            logger.info(
                f"⭐ {species_name} 结果: 准确率 {acc:.2%} | 总样本: {len(y_true)}"
            )

            # 保存单物种详细 CSV
            df_species = pd.DataFrame(details)
            df_species.to_csv(
                RESULT_DIR / f"detail_{species_name}_pth12_letterbox.csv",
                index=False,
                encoding="utf-8-sig",
            )

            # 生成单物种混淆矩阵
            save_confusion_matrix(y_true, y_pred, species_name)

            all_summary.append(
                {
                    "Species": species_name,
                    "Accuracy": f"{acc:.4f}",
                    "Sample_Size": len(y_true),
                }
            )

    # ========================== 4. 输出最终总结 ==========================
    if all_summary:
        # 1. 保存物种对比摘要 CSV
        summary_df = pd.DataFrame(all_summary)
        summary_df.to_csv(
            RESULT_DIR / "cross_species_summary_pth12_letterbox.csv", index=False
        )

        # 2. ✨ 生成并保存【全局总混淆矩阵】
        if overall_y_true:
            save_confusion_matrix(overall_y_true, overall_y_pred, "OVERALL_ALL_SPECIES")
            total_acc = accuracy_score(overall_y_true, overall_y_pred)

            # 3. ✨ 日志输出全局统计分数
            logger.info("\n" + "🏁" * 20)
            logger.info(f"🏆 跨物种实战测试最终报告 (pth12)")
            logger.info(f"📊 总评估样本量: {len(overall_y_true)}")
            logger.info(f"🎯 全局平均准确率: {total_acc:.2%}")
            logger.info("🏁" * 20)

        # 打印各作物得分榜
        logger.info("\n" + "=" * 50)
        logger.info("📋 跨物种得分明细：")
        logger.info("\n" + summary_df.to_string(index=False))
        logger.info("=" * 50)
        logger.info(
            f"✅ 评估任务圆满完成！所有报告存放在: {RESULT_DIR.relative_to(ROOT_DIR)}"
        )


if __name__ == "__main__":
    run_benchmark()
