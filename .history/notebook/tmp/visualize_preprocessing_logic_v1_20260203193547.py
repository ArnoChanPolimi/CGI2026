import os
import sys
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import numpy as np

# 设置路径
current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# 创建输出目录
OUTPUT_DIR = ROOT_DIR / "output" / "visual_debug"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 目标图片路径
IMG_PATH = (
    ROOT_DIR
    / "RAW_DATA"
    / "06_Jackfruit"
    / "biotic"
    / "Leaf_Miner"
    / "Leaf_Miner_5.jpg"
)


def save_tensor_as_image(tensor, name):
    """将归一化后的张量还原为可见图片并保存"""
    # 逆归一化 (Normalize 的反操作)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    # 限制范围在 0-1
    img_np = tensor.clamp(0, 1).numpy().transpose(1, 2, 0)
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    img_pil.save(OUTPUT_DIR / f"{name}.png")
    print(f"✅ Saved: {name}.png")


def visualize_steps():
    if not IMG_PATH.exists():
        print(f"❌ 找不到图片: {IMG_PATH}")
        return

    # 1. 原始图片 (Original)
    img = Image.open(IMG_PATH).convert("RGB")
    img.save(OUTPUT_DIR / "step1_original.png")
    print(f"✅ Saved: step1_original.png | Size: {img.size}")

    # 2. 等比缩放 (Resize long side to 256)
    target_size = 256
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = F.resize(img, (new_h, new_w))
    img_resized.save(OUTPUT_DIR / "step2_resized_256.png")
    print(f"✅ Saved: step2_resized_256.png | Size: {img_resized.size}")

    # 3. 补齐黑边 (Letterbox Padding to 256x256)
    delta_w = target_size - new_w
    delta_h = target_size - new_h
    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - (delta_w // 2),
        delta_h - (delta_h // 2),
    )
    img_padded = F.pad(img_resized, padding, fill=0)
    img_padded.save(OUTPUT_DIR / "step3_padded_square.png")
    print(f"✅ Saved: step3_padded_square.png | Size: {img_padded.size}")

    # 4. 中心截取 (Center Crop to 224)
    img_cropped = F.center_crop(img_padded, (224, 224))
    img_cropped.save(OUTPUT_DIR / "step4_cropped_224.png")
    print(f"✅ Saved: step4_cropped_224.png | Size: {img_cropped.size}")

    # 5. 颜色增强与标准化 (Color/Tensor Processing)
    # 这里模拟最终输入模型的张量样式（包含标准化后的视觉色偏）
    final_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tensor_img = final_transform(img_cropped)
    save_tensor_as_image(tensor_img, "step5_model_input_visual")

    print("\n🚀 所有可视化图片已保存在: output/visual_debug/")


if __name__ == "__main__":
    visualize_steps()
