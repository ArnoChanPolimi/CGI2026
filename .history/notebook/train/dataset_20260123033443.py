# notebook\train\dataset.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path


class PlantDataset(Dataset):
    def __init__(self, csv_path, root_dir, split="train", transform=None):
        """
        Args:
            csv_path: CSV 文件的相对路径
            root_dir: 项目根目录的 Path 对象
            split: 'train' 或 'test'
            transform: 图像预处理逻辑
        """
        df = pd.read_csv(csv_path)
        # 根据 split 过滤数据
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. 获取相对路径并拼成绝对路径
        img_rel_path = self.data.iloc[idx]["rel_path"]
        img_path = self.root_dir / img_rel_path

        # 2. 读取图片并转为 RGB
        image = Image.open(img_path).convert("RGB")
        label = int(self.data.iloc[idx]["label"])

        # 3. 应用预处理
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms():
    # 训练集：加入随机增强，让大脑更灵活
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 测试集：不折腾，只缩放和标准化，模拟真实考试
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, test_transform
