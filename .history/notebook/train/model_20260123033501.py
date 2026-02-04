# notebook\train\model.py
import torch.nn as nn
from torchvision import models


def get_plant_model(num_classes=2):
    # 使用目前最稳的 ResNet50 权重
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # 替换最后的全连接层，适配你的二分类任务 (Biotic/Abiotic)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model
