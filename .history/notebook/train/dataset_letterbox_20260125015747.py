# notebook\train\dataset_letterbox.py
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F


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


# 定义“长边压缩+填充”的逻辑函数
class LetterboxResize:
    def __init__(self, target_size=256):
        self.target_size = target_size

    def __call__(self, img):
        # 1. 逻辑：长边缩放到 256，短边等比缩放
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = F.resize(img, (new_h, new_w))

        # 2. 逻辑：填充黑边成 256x256 正方形
        # 计算上下左右需要补多少像素
        delta_w = self.target_size - new_w
        delta_h = self.target_size - new_h
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - (delta_w // 2),
            delta_h - (delta_h // 2),
        )
        return F.pad(img, padding, fill=0)  # fill=0 就是补黑边


def get_letterbox_transforms():
    train_tf = transforms.Compose(
        [
            # --- 你的三步走逻辑 ---
            LetterboxResize(256),  # 第一步 & 第二步：等比缩放并填充成 256
            transforms.RandomCrop(224),  # 第三步：在 256 里随机位移裁剪 224
            # --- 之前的旋转翻转逻辑 ---
            transforms.RandomChoice(
                [
                    transforms.RandomRotation((0, 0)),
                    transforms.RandomRotation((90, 90)),
                    transforms.RandomRotation((180, 180)),
                    transforms.RandomRotation((270, 270)),
                ]
            ),
            transforms.RandomHorizontalFlip(),
            # --- 标准化输出 ---
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_tf = transforms.Compose(
        [
            LetterboxResize(256),
            transforms.CenterCrop(224),  # 测试时固定切中心，不随机
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return train_tf, test_tf
