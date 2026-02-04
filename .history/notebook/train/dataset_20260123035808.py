# notebook/train/dataset.py
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PlantDataset(Dataset):
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


def get_transforms():
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            # 【新增逻辑】随机旋转 90, 180, 270 度
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
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, test_tf
