import torch
import pandas as pd
import sys
from pathlib import Path
from torch.utils.data import DataLoader

# 自动处理路径，确保能找到相关的自定义模块
current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from notebook.train.dataset import PlantDataset, get_transforms
from notebook.train.model import get_plant_model
from notebook.tools.logger_utils import get_logger

# ================= 配置路径 =================
SAVE_DIR = ROOT_DIR / "output" / "All_crop_train"
# 输出路径：output\All_crop_train\error_analysis
ERROR_OUT_DIR = SAVE_DIR / "error_analysis"
ERROR_OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = ROOT_DIR / "output" / "dataset_index" / "dataset_index.csv"
MODEL_PATH = SAVE_DIR / "checkpoint_epoch_7.pth"
ERROR_CSV_FILE = ERROR_OUT_DIR / "all_misclassified_cases.csv"
DEVICE = torch.device("cpu")

# 日志存放
logger = get_logger(log_dir=str(ERROR_OUT_DIR), log_filename="error_scan.log")


def export_all_errors():
    logger.info("--- 🔍 开始扫描测试集全量错题 ---")

    # 1. 加载模型
    if not MODEL_PATH.exists():
        logger.error(f"❌ 未找到权重文件: {MODEL_PATH.name}")
        return

    model = get_plant_model(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. 准备数据
    _, test_tf = get_transforms()
    test_ds = PlantDataset(CSV_PATH, ROOT_DIR, split="test", transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    # 标签映射
    label_map = {0: "Abiotic", 1: "Biotic"}

    # 获取测试集对应的原始数据
    full_df = test_ds.df
    test_meta = full_df[full_df["split"] == "test"].reset_index(drop=True)

    error_list = []

    logger.info(f"正在分析 {len(test_loader)} 张测试图片...")

    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            outputs = model(image.to(DEVICE))

            # 计算置信度
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

            actual_idx = label.item()
            pred_idx = pred.item()

            # 只要预测错了就记录
            if actual_idx != pred_idx:
                row_info = test_meta.iloc[i]

                error_list.append(
                    {
                        "FileName": Path(row_info["path"]).name,
                        "Relative_Path": row_info["path"],
                        "Ground_Truth": label_map[actual_idx],  # 应该是
                        "Model_Prediction": label_map[pred_idx],  # 被当作是
                        "Confidence": f"{confidence.item():.4f}",  # 模型确信度
                    }
                )

    # 3. 保存结果
    if error_list:
        error_df = pd.DataFrame(error_list)
        error_df.to_csv(ERROR_CSV_FILE, index=False, encoding="utf-8-sig")

        # 统计输出到日志
        bi_as_ab = len(
            error_df[
                (error_df["Ground_Truth"] == "Biotic")
                & (error_df["Model_Prediction"] == "Abiotic")
            ]
        )
        ab_as_bi = len(
            error_df[
                (error_df["Ground_Truth"] == "Abiotic")
                & (error_df["Model_Prediction"] == "Biotic")
            ]
        )

        logger.info(f"📊 统计结果:")
        logger.info(f"   - 应该是 Biotic 但被当作 Abiotic: {bi_as_ab} 张")
        logger.info(f"   - 应该是 Abiotic 但被当作 Biotic: {ab_as_bi} 张")
        logger.info(f"✅ 完整清单已保存至: {ERROR_CSV_FILE}")
    else:
        logger.info("🎉 完美！测试集无任何分类错误。")


if __name__ == "__main__":
    export_all_errors()
