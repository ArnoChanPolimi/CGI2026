# notebook\train\dataset_logic_v1.py
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PlantDatasetLogic(Dataset):
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


def get_logic_transforms():
    """
    最合理的逻辑：
    1. Resize(256) -> 等比例缩放，短边对齐到256，长边按比例缩放，不挤压变形。
    2. CenterCrop(224) -> 从中间切出224的正方形，保证叶片主体在视野内且比例真实。
    """
    train_tf = transforms.Compose(
        [
            transforms.Resize(256),  # 关键：只传一个数字，保持长宽比
            transforms.CenterCrop(224),
            transforms.RandomChoice(
                [
                    transforms.RandomRotation((0, 0)),
                    transforms.RandomRotation((90, 90)),
                    transforms.RandomRotation((180, 180)),
                    transforms.RandomRotation((270, 270)),
                ]
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, test_tf
