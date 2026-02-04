import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 确保导入逻辑版
sys.path.append(str(Path(__file__).resolve().parent.parent))
from train.dataset_logic_v1 import get_logic_transforms as get_transforms
from train.model import get_plant_model

# ========================== 1. 路径与配置 ==========================
current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parent.parent.parent
OUTPUT_DIR = ROOT_DIR / "output" / "visualize"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = (
    ROOT_DIR
    / "output"
    / "All_crop_train_logic_v1"
    / "checkpoint_epoch_letterbox_12.pth"
)
CSV_PATH = ROOT_DIR / "output" / "dataset_index" / "dataset_index_letterbox_v1.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_RELATIVE_PATHS_TO_ANALYZE = [
    # 01_Chilli (辣椒)
    "RAW_DATA/01_Chilli/abiotic/Nutrition_Deficiency/Nutrition Deficiency_1082.png",  # train
    "RAW_DATA/01_Chilli/abiotic/Nutrition_Deficiency/Nutrition Deficiency_1094.png",  # test true positive
    "RAW_DATA/01_Chilli/abiotic/Nutrition_Deficiency/image_851.jpg",  # 错误
    "RAW_DATA/01_Chilli/abiotic/Chili Healthy Leaf/Healthy Leaf00061.JPG",  # 误报
    # 02_Tomato (番茄)
    "RAW_DATA/02_Tomato/abiotic/Magnesium Deficiency/Magnesium Deficiency_26.jpg",  # train
    "RAW_DATA/02_Tomato/abiotic/Magnesium Deficiency/Magnesium Deficiency_30.jpg",  # test true positive
    "RAW_DATA/02_Tomato/abiotic/Magnesium Deficiency/Magnesium Deficiency_113.jpg",  # 错误
    "RAW_DATA/02_Tomato/abiotic/Healthy/Healthy_15.jpg",  # 误报
    # 03_Carambola (杨桃)
    "RAW_DATA/03_Carambola/biotic/Insect_past/Augmented_0_8968.jpeg",  # train
    "RAW_DATA/03_Carambola/biotic/Insect_past/Augmented_0_9278.jpeg",  # true positive
    "RAW_DATA/03_Carambola/abiotic/Healthy/Augmented_0_1941.jpeg",  # false negative
    "RAW_DATA/03_Carambola/abiotic/Iron Deficiency/Augmented_0_1614.jpeg",  # false positive
    "RAW_DATA/03_Carambola/biotic/Algal_Leaf_Spot/Augmented_0_1337.jpeg",  # false negative
    # 04_Guava (番石榴)
    "RAW_DATA/04_Guava/abiotic/Nutritional Deficiency/IMG20230817155123.jpg",  # false negative
    # 05_cherry (樱桃)
    "RAW_DATA/05_cherry/biotic/Chewing Insects/2025_03_06_14_29_IMG_3303.jpg",  # false negative
    # 06_Jackfruit (菠萝蜜)
    "RAW_DATA/06_Jackfruit/abiotic/Healthy/Healthy_447.jpg",  # false negative
    "RAW_DATA/06_Jackfruit/abiotic/Senescence/Senescence_20.jpg",
    # 07_Mazie (玉米)
    "RAW_DATA/07_Mazie/abiotic/Nitrogen/2 (58).jpeg",  # false negative
    "RAW_DATA/07_Mazie/biotic/Corn_leaf_blight/train_Corn leaf blight_86.jpg",
    # 08_Wheat (小麦)
    "RAW_DATA/08_Wheat/abiotic/Ndeficient/144.jpg",  # false negative
    "RAW_DATA/08_Wheat/biotic/WheatLeafRust/train_104.jpg",
    # 09_AloeVera (库拉索芦荟)
    "RAW_DATA/09_AloeVera/abiotic/Healthy/Augmented_0_1168.jpeg",  # false negative
    "RAW_DATA/09_AloeVera/abiotic/Sunburn/Augmented_0_4249.jpeg",  # false negative
    "RAW_DATA/09_AloeVera/biotic/Aloe Rust/Augmented_0_5155.jpeg",  # false negative
]

# ========================== 2. Grad-CAM 核心逻辑 ==========================


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

        # 使用更稳健的 Hook
        self.target_layer.register_forward_hook(self._save_feature_maps)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_feature_maps(self, module, input, output):
        self.feature_maps = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        loss = output[0, class_idx]
        loss.backward()

        # 核心修复 1: 使用 detach() 剥离梯度
        gradients = self.gradients.detach()
        feature_maps = self.feature_maps.detach()

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze().cpu().numpy()


# ========================== 3. 可视化函数 ==========================


def visualize_grad_cam(
    model, target_layer, image_path, preprocess_transform, output_dir, pth_name
):
    # 加载图片
    img_pil = mpimg.imread(image_path)
    if img_pil.ndim == 2:
        img_pil = np.stack([img_pil] * 3, axis=-1)

    # 统一转为 uint8 格式的 BGR (OpenCV 标准)
    if img_pil.max() <= 1.0:
        img_pil = (img_pil * 255).astype(np.uint8)
    original_img = cv2.cvtColor(img_pil, cv2.COLOR_RGB2BGR)

    input_tensor = preprocess_transform(original_img).unsqueeze(0).to(DEVICE)

    cam_extractor = GradCAM(model, target_layer)
    cam = cam_extractor(input_tensor)

    # 调整大小
    cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))

    # 核心修复 2: 强制转换为 uint8 确保 addWeighted 不报错
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = heatmap.astype(np.uint8)
    original_img = original_img.astype(np.uint8)

    # 叠加
    overlay_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    # 推理结果
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        conf = probs[pred_idx].item() * 100

    class_names = {0: "Abiotic", 1: "Biotic"}
    pred_name = class_names.get(pred_idx, "Unknown")

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Pred: {pred_name} ({conf:.1f}%)")
    axes[1].axis("off")

    out_path = output_dir / f"{image_path.stem}_{pth_name}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"✅ 保存: {out_path.name}")


# ========================== 4. 执行 ==========================

if __name__ == "__main__":
    model = get_plant_model(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # --- 把这两行加在这里，紧跟着 model 定义完之后 ---
    print("-" * 50)
    print(f"✅ 证实：这个 model.features 列表里一共有: {len(model.features)} 个大层")
    print(model.features)
    print("-" * 50)
    print(f"--- 现在的运行设备是: {DEVICE} ---")
    # ----------------------------------------------

    # 寻找最后一层卷积
    target_layer = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    preprocess = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    pth_tag = MODEL_PATH.stem
    for rel_path in tqdm(IMAGE_RELATIVE_PATHS_TO_ANALYZE):
        full_path = ROOT_DIR / rel_path
        if full_path.exists():
            visualize_grad_cam(
                model, target_layer, full_path, preprocess, OUTPUT_DIR, pth_tag
            )
