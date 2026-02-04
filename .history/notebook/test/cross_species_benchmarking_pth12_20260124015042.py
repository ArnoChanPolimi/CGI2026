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
from torch.utils.data import DataLoader

# ========================== 1. 动态路径与环境配置 ==========================
current_file = Path(__file__).resolve()
# 脚本在 notebook\test\，向上退两级到达项目根目录
ROOT_DIR = current_file.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from dataset_logic_v1 import get_logic_transforms
from model import get_plant_model
from notebook.tools.logger_utils import get_logger

# 相对路径配置
MODEL_PATH = ROOT_DIR / "output" / "All_crop_train_logic_v1" / "checkpoint_epoch_12.pth"
OTHER_DATA_DIR = ROOT_DIR / "other"  # 使用相对于根目录的 other 文件夹
RESULT_DIR = ROOT_DIR / "output" / "All_crop_train_logic_v1" / "cross_species_test"
LOG_DIR = ROOT_DIR / "log" / "cross_species_test"

# 创建必要目录
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 初始化日志
logger = get_logger(
    log_dir=str(LOG_DIR), log_filename="cross_species_benchmarking_pth12.log"
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
    """为每个物种生成独立的混淆矩阵图"""
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
    save_path = RESULT_DIR / f"cm_{species_name}.png"
    plt.savefig(save_path)
    plt.close()
    logger.info(f"📊 混淆矩阵已保存至: {save_path.name}")


# ========================== 3. 执行评估逻辑 ==========================
def run_benchmark():
    all_summary = []

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
                            "SubCategory": img_p.parent.name,  # 记录具体的病害名(如 Canker)
                        }
                    )
                except Exception as e:
                    logger.error(f"❌ 处理图片出错 {img_p.name}: {e}")

        # 统计单物种结果
        if y_true:
            acc = accuracy_score(y_true, y_pred)
            logger.info(
                f"⭐ {species_name} 结果: 准确率 {acc:.2%} | 总样本: {len(y_true)}"
            )

            # 保存错题集 CSV
            df_species = pd.DataFrame(details)
            df_species.to_csv(
                RESULT_DIR / f"detail_{species_name}.csv",
                index=False,
                encoding="utf-8-sig",
            )

            # 生成混淆矩阵
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
        summary_df = pd.DataFrame(all_summary)
        summary_df.to_csv(RESULT_DIR / "cross_species_summary.csv", index=False)
        logger.info("\n" + "=" * 50)
        logger.info("🏆 跨物种实战测试总表：")
        logger.info("\n" + summary_df.to_string(index=False))
        logger.info("=" * 50)
        logger.info(f"✅ 所有评估任务圆满完成！报告存放在: {RESULT_DIR}")


if __name__ == "__main__":
    run_benchmark()
