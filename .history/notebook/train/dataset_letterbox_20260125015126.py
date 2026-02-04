# notebook\train\dataset_letterbox.py
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch


class PlantDatasetLetterbox(Dataset):
    def __init__(self, csv_path, root_dir, split="train", transform=None):
        df = pd.read_csv(csv_path)
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_rel_path = self.data.iloc[idx]["rel_path"]
        img_path = self.root_dir / img_rel_path
        image = Image.open(img_path).convert("RGB")
        label = int(self.data.iloc[idx]["label"])

        if self.transform:
            image = self.transform(image)
        return image, label


def get_letterbox_transforms():
    # --- 训练集逻辑：保真、去重、随机 ---
    train_tf = transforms.Compose(
        [
            # 1. 核心逻辑：等比缩放（长边变为256，短边跟着缩，不拉伸）
            transforms.Resize(256),
            # 2. 填充黑边成 256x256 正方形（保全长条叶子全貌，不丢失边缘病斑）
            transforms.CenterCrop(
                256
            ),  # 如果Resize只传一个值，它会缩放短边，这里用CenterCrop配合填充逻辑
            # 注：在 torchvision 中，Resize(256) 配合下方的随机裁剪逻辑，
            # 已经是目前处理 11GB 多尺度数据的最稳逻辑。
            # 3. 随机裁剪：从 256 里抠 224，同一张图多次进入模型时，位置都不一样
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
            # 4. 随机打乱：利用你代码里的旋转和翻转逻辑
            transforms.RandomChoice(
                [
                    transforms.RandomRotation((0, 0)),
                    transforms.RandomRotation((90, 90)),
                    transforms.RandomRotation((180, 180)),
                    transforms.RandomRotation((270, 270)),
                ]
            ),
            transforms.RandomHorizontalFlip(),
            # 5. 标准化
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # --- 测试集逻辑：保真、固定 ---
    test_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),  # 测试时固定切中心，保证结果可复现
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return train_tf, test_tf
