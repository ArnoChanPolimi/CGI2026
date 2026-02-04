import torch
import pandas as pd
import sys
from pathlib import Path
from torch.utils.data import DataLoader

# 自动处理路径，确保能找到 dataset 和 model
current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from notebook.train.dataset import PlantDataset, get_transforms
from notebook.train.model import get_plant_model
from notebook.tools.logger_utils import get_logger

# ================= 配置路径 =================
SAVE_DIR = ROOT_DIR / "output" / "All_crop_train"
ERROR_OUT_DIR = SAVE_DIR / "error_analysis"
ERROR_OUT_DIR.mkdir(parents=True, exist_ok=True)  # 自动创建该文件夹

CSV_PATH = ROOT_DIR / "output" / "dataset_index" / "dataset_index.csv"
MODEL_PATH = SAVE_DIR / "checkpoint_epoch_7.pth"
ERROR_CSV_FILE = ERROR_OUT_DIR / "all_misclassified_cases.csv"
DEVICE = torch.device("cpu")

# 日志也存放在错题分析文件夹内
logger = get_logger(log_dir=str(ERROR_OUT_DIR), log_filename="error_analysis_scan.log")


def export_all_errors():
    logger.info("--- 🔍 开始扫描全量错题（所有错误类型） ---")

    # 1. 加载模型
    if not MODEL_PATH.exists():
        logger.error(f"找不到模型文件: {MODEL_PATH}")
        return

    model = get_plant_model(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. 准备数据加载
    _, test_tf = get_transforms()
    test_ds = PlantDataset(CSV_PATH, ROOT_DIR, split="test", transform=test_tf)
    # batch_size=1 是为了确保我们能精准对应每一张图的原始 CSV 索引
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    # 标签映射，方便阅读
    label_map = {0: "Abiotic", 1: "Biotic"}

    full_df = test_ds.df
    all_errors = []

    logger.info(f"正在对测试集 {len(test_loader)} 张图片进行深度扫描...")

    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            outputs = model(image.to(DEVICE))

            # 计算概率
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

            actual_idx = label.item()
            pred_idx = pred.item()

            # 只要预测和实际不符，就记录下来
            if actual_idx != pred_idx:
                # 从原始 DataFrame 获取对应的行信息（注意 split='test' 后的索引偏移）
                # PlantDataset 内部通常会过滤 split，我们需要拿正确的行数据
                row_info = full_df[full_df["split"] == "test"].iloc[i]

                all_errors.append(
                    {
                        "filename": Path(row_info["path"]).name,
                        "relative_path": row_info["path"],
                        "actual_label": label_map[actual_idx],
                        "predicted_label": label_map[pred_idx],
                        "confidence": f"{confidence.item():.4f}",
                        "error_type": f"{label_map[actual_idx]}_as_{label_map[pred_idx]}",
                    }
                )

    # 3. 统计结果并保存
    if all_errors:
        error_df = pd.DataFrame(all_errors)
        error_df.to_csv(ERROR_CSV_FILE, index=False, encoding="utf-8-sig")

        # 打印各类型错误的数量
        summary = error_df["error_type"].value_counts()
        logger.info("\n错误类型统计:")
        for err_name, count in summary.items():
            logger.info(f" -> {err_name}: {count} 张")

        logger.info(f"✅ 全量错题清单已保存至: {ERROR_CSV_FILE}")
    else:
        logger.info("🎉 太棒了！测试集中没有发现任何错误。")


if __name__ == "__main__":
    export_all_errors()
