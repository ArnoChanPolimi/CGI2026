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
# 1. 核心变换逻辑 (保持长边填充)
# ==========================================
class LongSidePadTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        res = ImageOps.contain(img, (self.size, self.size))
        return ImageOps.pad(res, (self.size, self.size), color=0)


def setup_logger(log_dir):
    # 创建日志目录
    log_dir.mkdir(parents=True, exist_ok=True)
    # 日志文件命名为 test_run.txt
    log_file = log_dir / "test_run_log.txt"

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
    # 路径回退：notebook/test_pepper_on_banana.py -> notebook -> 根目录
    project_root = script_path.parents[1]

    # 输入路径
    data_dir = project_root / "data" / "banana_challenge"
    model_path = project_root / "output" / "pepper_model_longside.pth"

    # 输出路径
    output_dir = project_root / "output" / "pepper_challenge_banana"
    log_folder = project_root / "log" / "pepper_test_banana"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化日志
    logger = setup_logger(log_folder)

    device = torch.device("cpu")
    logger.info("=" * 50)
    logger.info("🚀 启动跨物种挑战：辣椒模型检测香蕉病害")
    logger.info(f"📍 测试数据目录: {data_dir}")
    logger.info(f"🧠 使用模型文件: {model_path}")
    logger.info(f"📝 日志已记录至: {log_folder}")

    # --- 2. 数据加载 ---
    test_transform = transforms.Compose(
        [
            LongSidePadTransform(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if not data_dir.exists():
        logger.error(
            "❌ 错误：找不到 'data/banana_challenge' 文件夹。请先运行预处理脚本！"
        )
        return

    try:
        test_dataset = datasets.ImageFolder(data_dir, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=16, shuffle=False
        )
        class_names = test_dataset.classes
        logger.info(
            f"📦 成功加载香蕉样本。类别: {class_names}, 样本总数: {len(test_dataset)}"
        )
    except Exception as e:
        logger.error(f"❌ 加载数据集失败: {e}")
        return

    # --- 3. 加载大脑模型 ---
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    if not model_path.exists():
        logger.error(f"❌ 找不到模型文件: {model_path}")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info("✅ 辣椒 2分类模型加载完成。")
    except Exception as e:
        logger.error(f"❌ 模型权重解析失败: {e}")
        return

    model.to(device).eval()

    # --- 4. 执行推理 ---
    y_true = []
    y_pred = []

    logger.info("🔍 正在提取香蕉叶片特征并对比辣椒病理模式...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    # --- 5. 保存结果 ---
    # 分类报告
    report = classification_report(y_true, y_pred, target_names=class_names)
    report_file = output_dir / "challenge_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=== Pepper-to-Banana Transfer Test Report ===\n")
        f.write(report)
    logger.info(f"📊 评估报告已生成: {report_file}")

    # 混淆矩阵
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.YlGn, values_format="d")
    plt.title("Cross-Species Challenge: Pepper Model on Banana")

    cm_file = output_dir / "challenge_confusion_matrix.png"
    plt.savefig(cm_file)
    plt.close()

    logger.info(f"🖼️ 混淆矩阵图已保存: {cm_file}")
    logger.info("🎉 所有测试流程已完成。")
    logger.info("=" * 50)


if __name__ == "__main__":
    run_challenge()
