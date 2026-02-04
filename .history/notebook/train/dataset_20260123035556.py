# notebook/train/dataset.py
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PlantDataset(Dataset):
    def __init__(self, csv_path, root_dir, split="train", transform=None):
        df = pd.read_csv(csv_path)
        # 只筛选对应的训练集或测试集
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 拼凑绝对路径
        img_rel_path = self.data.iloc[idx]["rel_path"]
        img_path = self.root_dir / img_rel_path

        image = Image.open(img_path).convert("RGB")
        label = int(self.data.iloc[idx]["label"])

        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms():
    # 针对 V3 的标准化参数（来自 ImageNet 数据集）
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),  # 数据增强：水平翻转
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, test_tf
