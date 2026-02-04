import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os
import logging
from pathlib import Path


# ==========================================
# 1. 核心变换逻辑：必须与训练时完全一致
# ==========================================
class LongSidePadTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # 保持比例缩放
        res = ImageOps.contain(img, (self.size, self.size))
        # 填充黑色背景补齐为正方形
        return ImageOps.pad(res, (self.size, self.size), color=0)


def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger()


def run_challenge():
    # --- 路径自动定位 ---
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]  # notebook 文件夹的上一级即根目录

    # 输入路径
    data_dir = project_root / "data" / "banana_challenge"
    model_path = project_root / "output" / "pepper_model_longside.pth"

    # 输出路径
    output_dir = project_root / "output" / "pepper_challenge_banana"
    log_dir = project_root / "log"

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_dir / "pepper_test_banana.log")

    device = torch.device("cpu")  # 强制使用CPU，确保兼容性
    logger.info("🚀 开始跨物种挑战：使用辣椒模型识别香蕉病害")
    logger.info(f"📍 测试数据: {data_dir}")
    logger.info(f"🧠 模型文件: {model_path}")

    # --- 2. 数据加载与预处理 ---
    test_transform = transforms.Compose(
        [
            LongSidePadTransform(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if not data_dir.exists():
        logger.error("❌ 找不到挑战数据集，请先运行预处理脚本！")
        return

    test_dataset = datasets.ImageFolder(data_dir, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False
    )
    class_names = test_dataset.classes
    logger.info(f"📦 成功加载数据，类别: {class_names}, 总数: {len(test_dataset)}")

    # --- 3. 加载训练好的大脑 ---
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info("✅ 辣椒大脑模型加载成功！")
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {str(e)}")
        return

    model.to(device).eval()

    # --- 4. 批量推理 ---
    y_true = []
    y_pred = []

    logger.info("🔍 正在分析香蕉图像特征...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    # --- 5. 生成报告与结果 ---
    # A. 分类报告
    report = classification_report(y_true, y_pred, target_names=class_names)
    report_file = output_dir / "challenge_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=== Pepper Model on Banana Challenge ===\n")
        f.write(report)
    logger.info(f"📊 分类报告已保存: {report_file}")
    print("\n" + report)

    # B. 混淆矩阵绘图
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Greens, values_format="d")
    plt.title("Confusion Matrix: Pepper Model identifying Banana")

    cm_file = output_dir / "challenge_confusion_matrix.png"
    plt.savefig(cm_file)
    plt.close()
    logger.info(f"🖼️ 混淆矩阵图已保存: {cm_file}")
    logger.info("🎉 挑战测试圆满结束！")


if __name__ == "__main__":
    run_challenge()
