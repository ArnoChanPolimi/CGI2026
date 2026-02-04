# notebook/train/model.py
import torch.nn as nn
from torchvision import models


def get_plant_model(num_classes=2):
    # 1. 加载预训练的 MobileNetV3 大脑
    # 它会自动下载约 20MB 的权重文件
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

    # 2. 获取分类器最后一层的输入维度
    # MobileNetV3 的 classifier 是一个序列，第4层（索引3）是最终的线性输出层
    num_ftrs = model.classifier[3].in_features

    # 3. 将输出按钮改为 2（Biotic 和 Abiotic）
    model.classifier[3] = nn.Linear(num_ftrs, num_classes)

    return model
