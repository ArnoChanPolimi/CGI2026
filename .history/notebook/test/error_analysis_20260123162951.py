# notebook\test\error_analysis.py
import torch
import pandas as pd
import sys
from pathlib import Path
from torch.utils.data import DataLoader

# 自动处理路径
current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from notebook.train.dataset import PlantDataset, get_transforms
from notebook.train.model import get_plant_model
from notebook.tools.logger_utils import get_logger

# ================= 配置路径 =================
SAVE_DIR = ROOT_DIR / "output" / "All_crop_train"
ERROR_OUT_DIR = SAVE_DIR / "error_analysis"
ERROR_OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = ROOT_DIR / "output" / "dataset_index" / "dataset_index.csv"
MODEL_PATH = SAVE_DIR / "checkpoint_epoch_7.pth"
ERROR_CSV_FILE = ERROR_OUT_DIR / "all_misclassified_cases.csv"
DEVICE = torch.device("cpu")

logger = get_logger(log_dir=str(ERROR_OUT_DIR), log_filename="error_scan.log")


def export_all_errors():
    logger.info("--- 🔍 开始扫描测试集全量错题 ---")

    if not MODEL_PATH.exists():
        logger.error(f"❌ 未找到权重文件: {MODEL_PATH.name}")
        return

    # 1. 既然 Dataset 里没存 df，我们在这里亲自动手读
    full_df = pd.read_csv(CSV_PATH)
    test_meta = full_df[full_df["split"] == "test"].reset_index(drop=True)

    # 2. 加载模型
    model = get_plant_model(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 3. 准备加载器
    _, test_tf = get_transforms()
    test_ds = PlantDataset(CSV_PATH, ROOT_DIR, split="test", transform=test_tf)
    # 严格使用 batch_size=1
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    label_map = {0: "Abiotic", 1: "Biotic"}
    error_list = []

    logger.info(f"正在分析 {len(test_loader)} 张测试图片...")

    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            outputs = model(image.to(DEVICE))

            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

            actual_idx = label.item()
            pred_idx = pred.item()

            if actual_idx != pred_idx:
                # 直接从我们刚才读的 test_meta 里拿数据
                row_info = test_meta.iloc[i]

                error_list.append(
                    {
                        "FileName": Path(row_info["path"]).name,
                        "Relative_Path": row_info["path"],
                        "Ground_Truth": label_map[actual_idx],
                        "Model_Prediction": label_map[pred_idx],
                        "Confidence": f"{confidence.item():.4f}",
                    }
                )

    # 4. 保存
    if error_list:
        error_df = pd.DataFrame(error_list)
        error_df.to_csv(ERROR_CSV_FILE, index=False, encoding="utf-8-sig")

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
