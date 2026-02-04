# notebook\tools\visualize_grad_cam.py
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from PIL import Image

# 确保导入逻辑版
sys.path.append(str(Path(__file__).resolve().parent.parent))
from train.model import get_plant_model


# ========================== 1. 核心修复：引入与训练完全一致的 Letterbox ==========================
class LetterboxResize:
    def __init__(self, target_size=256):
        self.target_size = target_size

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = F.resize(img, (new_h, new_w))
        delta_w = self.target_size - new_w
        delta_h = self.target_size - new_h
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - (delta_w // 2),
            delta_h - (delta_h // 2),
        )
        return F.pad(img, padding, fill=0)


# ========================== 2. Grad-CAM 工具类 ==========================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        self.handlers = [
            target_layer.register_forward_hook(self._save_feature_maps),
            target_layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_feature_maps(self, module, input, output):
        self.feature_maps = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        output[0, class_idx].backward()
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam.squeeze().cpu().numpy()

    def release(self):
        for h in self.handlers:
            h.remove()


# ========================== 3. 主程序 ==========================
def main():
    # 配置
    current_file = Path(__file__).resolve()
    ROOT_DIR = current_file.parent.parent.parent
    MODEL_PATH = (
        ROOT_DIR
        / "output"
        / "All_crop_train_logic_v1"
        / "checkpoint_epoch_letterbox_12.pth"
    )
    OUTPUT_DIR = ROOT_DIR / "output" / "visualize_fixed"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = get_plant_model(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 预处理流程 (必须和训练集 test_tf 一模一样)
    vis_transform = transforms.Compose(
        [LetterboxResize(256), transforms.CenterCrop(224)]
    )
    tensor_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    images_to_test = [
        # 01_Chilli (辣椒)
        "RAW_DATA/01_Chilli/abiotic/Nutrition_Deficiency/Nutrition Deficiency_1082.png",  # train
        "RAW_DATA/01_Chilli/abiotic/Nutrition_Deficiency/Nutrition Deficiency_1094.png",  # test true positive
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

    for rel_path in tqdm(images_to_test):
        img_full_path = ROOT_DIR / rel_path
        if not img_full_path.exists():
            continue

        # 1. 关键：获取“物理对齐”后的底图
        raw_pil = Image.open(img_full_path).convert("RGB")
        processed_pil = vis_transform(raw_pil)
        input_tensor = tensor_transform(processed_pil).unsqueeze(0).to(DEVICE)

        # 2. 推理结果
        with torch.enable_grad():  # Grad-CAM 需要梯度
            output = model(input_tensor)
            pred_idx = output.argmax(1).item()
            conf = torch.softmax(output, 1)[0, pred_idx].item()

            # 3. 提取两层特征 (Layer 0 看整体, Layer 16 看病斑)
            cam0_obj = GradCAM(model, model.features[0])
            mask0 = cam0_obj(input_tensor, pred_idx)
            cam0_obj.release()

            cam16_obj = GradCAM(model, model.features[16])
            mask16 = cam16_obj(input_tensor, pred_idx)
            cam16_obj.release()

        # 4. 绘图展示
        base_img = np.array(processed_pil)

        def get_res(mask, b_img):
            m_resized = cv2.resize(mask, (224, 224))
            heatmap = cv2.applyColorMap(np.uint8(255 * m_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            return cv2.addWeighted(b_img, 0.6, heatmap, 0.4, 0)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(base_img)
        axes[0].set_title(
            f"Input (Letterbox)\nPred: {'Biotic' if pred_idx==1 else 'Abiotic'} {conf:.1%}"
        )
        axes[1].imshow(get_res(mask0, base_img))
        axes[1].set_title("Layer 0 (Object Location)")
        axes[2].imshow(get_res(mask16, base_img))
        axes[2].set_title("Layer 16 (Disease Focus)")

        for ax in axes:
            ax.axis("off")
        plt.savefig(OUTPUT_DIR / f"fixed_{Path(rel_path).stem}.png")
        plt.close()


if __name__ == "__main__":
    main()
