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
import re

# ========================== 核心修复：路径补全 ==========================
# 1. 彻底人肉锁定项目根目录
current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parents[2]  # 强制跳 3 级：tools -> notebook -> CGI_project

# 2. 强制覆盖 sys.path，把最高优先级的路径塞进去
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "notebook"))

# 3. 干净导入，不从别的脚本借 ROOT_DIR
try:
    from train.model import get_plant_model

    print("✅ 成功导入 model 模块")
except ImportError:
    from notebook.train.model import get_plant_model

    print("✅ 通过备用路径导入 model 模块")

# ======================================================================


# ========================== 1. 核心修复：引入 Letterbox 逻辑 ==========================
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


# ========================== 2. Grad-CAM 工具类 (保持你原本能跑的版本) ==========================
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
    current_file = Path(__file__).resolve()
    ROOT_DIR = current_file.parent.parent.parent
    MODEL_PATH = (
        ROOT_DIR
        / "output"
        / "All_crop_train_logic_v1"
        / "pth"
        / "checkpoint_epoch_letterbox_8.pth"
    )
    # --- ✨ 2. 动态提取版本标签 (如 pth8) ---
    try:
        # 寻找文件名中的数字，例如从 'checkpoint_epoch_letterbox_8.pth' 提取 '8'
        all_numbers = re.findall(r"(\d+)", MODEL_PATH.name)
        epoch_val = all_numbers[-1] if all_numbers else "unknown"
        ver_tag = f"pth{epoch_val}"
    except Exception:
        ver_tag = "unknown"

    # --- ✨ 3. 结果输出目录 (分流至 output/visualize/pth8) ---
    OUTPUT_DIR = ROOT_DIR / "output" / "visualize" / ver_tag
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"🚀 正在使用模型 {ver_tag} 进行可视化...")
    print(f"📂 结果将保存至: {OUTPUT_DIR.relative_to(ROOT_DIR)}")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_plant_model(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # --- 修复 1: 预处理必须带 Letterbox ---
    preprocess_vis = transforms.Compose(
        [LetterboxResize(256), transforms.CenterCrop(224)]
    )
    preprocess_tensor = transforms.Compose(
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

        # --- 修复 2: 统一底图与 Tensor 的坐标系 ---
        raw_pil = Image.open(img_full_path).convert("RGB")
        processed_pil = preprocess_vis(raw_pil)
        input_tensor = preprocess_tensor(processed_pil).unsqueeze(0).to(DEVICE)

        with torch.enable_grad():
            # 获取 0 层 (浅层特征：轮廓/纹理)
            cam0_obj = GradCAM(model, model.features[0])
            mask0 = cam0_obj(input_tensor)
            cam0_obj.release()

            # --- 新增：获取 13 层 (中深层特征) ---
            cam13_obj = GradCAM(model, model.features[13])
            mask13 = cam13_obj(input_tensor)
            cam13_obj.release()

            # 获取 16 层 (深层特征：病斑语义)
            cam16_obj = GradCAM(model, model.features[16])
            mask16 = cam16_obj(input_tensor)
            cam16_obj.release()

        # --- 修复 3: 叠加与颜色修正 ---# --- 修复 3: 叠加与颜色修正 (改为 1行4列) ---
        base_img_rgb = np.array(processed_pil)

        def get_res(mask, b_img_rgb):
            m_resized = cv2.resize(mask, (224, 224))
            heatmap = cv2.applyColorMap(np.uint8(255 * m_resized), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            return cv2.addWeighted(b_img_rgb, 0.6, heatmap_rgb, 0.4, 0)

        # 改为 1行4列，figsize 宽度增加到 24 保持比例
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        # 第 1 列：原图
        axes[0].imshow(base_img_rgb)
        axes[0].set_title("Input (Letterboxed)")

        # 第 2 列：Layer 0
        axes[1].imshow(get_res(mask0, base_img_rgb))
        axes[1].set_title("Layer 0 (Edges)")

        # 第 3 列：Layer 13 (新增)
        axes[2].imshow(get_res(mask13, base_img_rgb))
        axes[2].set_title("Layer 13 (Mid-Features)")

        # 第 4 列：Layer 16
        axes[3].imshow(get_res(mask16, base_img_rgb))
        axes[3].set_title("Layer 16 (Disease)")

        for ax in axes:
            ax.axis("off")

        # --- ✨ 4. 合理的文件命名 ---
        # 命名格式：vis_pth8_病害名_原图名.png
        # 这样即使文件被移动，也能通过文件名知道是哪个版本出的图
        img_stem = Path(rel_path).stem
        save_name = f"vis_{ver_tag}_{img_stem}.png"

        # 保存路径
        save_path = OUTPUT_DIR / save_name
        plt.savefig(save_path, bbox_inches="tight")  # bbox_inches 确保边缘不留白
        plt.close()


if __name__ == "__main__":
    main()
