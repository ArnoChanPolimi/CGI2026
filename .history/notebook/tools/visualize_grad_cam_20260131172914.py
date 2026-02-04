# notebook\tools\visualize_grad_cam.py
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 确保导入逻辑版
# 这里的导入路径需要根据 visualize_grad_cam.py 相对于 dataset_logic_v1.py 的位置进行调整
# 假设 dataset_logic_v1.py 在 notebook\train\ 下，那么需要回溯两级
sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)  # 添加 notebook 目录到 sys.path
from train.dataset_logic_v1 import (
    PlantDatasetLogic as PlantDataset,
    get_logic_transforms as get_transforms,
)
from train.model import get_plant_model

# ========================== 1. 路径与配置 ==========================
current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parent.parent.parent  # 项目根目录，例如 D:\Aa_Polimi\...
OUTPUT_DIR = ROOT_DIR / "output" / "visualize"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # 创建输出目录

# --- 用户可配置项 ---
# 待分析的 .pth 模型路径
# 例如: ROOT_DIR / "output" / "All_crop_train_logic_v1" / "checkpoint_epoch_letterbox_12.pth"
MODEL_PATH = (
    ROOT_DIR
    / "output"
    / "All_crop_train_logic_v1"
    / "checkpoint_epoch_letterbox_12.pth"
)

# dataset_index.csv 的路径
CSV_PATH = ROOT_DIR / "output" / "dataset_index" / "dataset_index_letterbox_v1.csv"

# 设备配置 (没有独显用CPU，有独显用CUDA)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 你想可视化的图片相对路径列表
# 这些路径是相对于 ROOT_DIR 的，且应该在 dataset_index_letterbox_v1.csv 中存在
# 例如: "data/raw_data/train/Apple_Scab/image_1.jpg"
# 确保这些图片在你的训练/测试集中
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
    "RAW_DATA/04_Guava/abiotic/Nutritional Deficiency/IMG20230817155123.jpg",  # 请替换实际文件名
    # 05_cherry (樱桃)
    "RAW_DATA/05_cherry/biotic/Chewing Insects/2025_03_06_14_29_IMG_3303.jpg",  # 请替换实际文件名
    # 06_Jackfruit (菠萝蜜)
    "RAW_DATA/06_Jackfruit/abiotic/Healthy/Healthy_447.jpg",  # 请替换实际文件名
    "RAW_DATA/06_Jackfruit/abiotic/Senescence/Senescence_20.jpg",
    # 07_Mazie (玉米)
    "RAW_DATA/07_Mazie/abiotic/Nitrogen/2 (58).jpeg",  # 请替换实际文件名
    "RAW_DATA/07_Mazie/biotic/Corn_leaf_blight/train_Corn leaf blight_86.jpg",
    # 08_Wheat (小麦)
    "RAW_DATA/08_Wheat/abiotic/Ndeficient/144.jpg",  # 请替换实际文件名
    "RAW_DATA/08_Wheat/biotic/WheatLeafRust/train_104.jpg",
    # 09_AloeVera (库拉索芦荟)
    "RAW_DATA/09_AloeVera/abiotic/Healthy/Augmented_0_1168.jpeg",  # 请替换实际文件名
    "RAW_DATA/09_AloeVera/abiotic/Sunburn/Augmented_0_4249.jpeg",  # 请替换实际文件名
    "RAW_DATA/09_AloeVera/biotic/Aloe Rust/Augmented_0_5155.jpeg",  # 请替换实际文件名
]

# ========================== 2. Grad-CAM 核心逻辑 ==========================


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.model.eval()
        self.feature_maps = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._save_feature_maps)
        self.target_layer.register_backward_hook(self._save_gradients)

    def _save_feature_maps(self, module, input, output):
        self.feature_maps = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)  # 前向传播

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()  # 预测分数最高的类别

        target_one_hot = torch.zeros_like(output)
        target_one_hot[:, class_idx] = 1  # 为目标类别创建 One-Hot 向量

        output.backward(
            gradient=target_one_hot, retain_graph=True
        )  # 反向传播，计算梯度

        gradients = self.gradients  # 获取目标层梯度
        feature_maps = self.feature_maps  # 获取目标层特征图

        # 计算权重 (全局平均池化梯度)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # 将权重与特征图相乘，并ReLU激活
        cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
        cam = torch.relu(cam)

        # 归一化到 0-1 范围
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)  # 加一个 epsilon 防止除零

        return cam.squeeze().cpu().numpy()


# ========================== 3. 可视化函数 ==========================


def visualize_grad_cam(
    model,
    target_layer,
    image_path,
    original_transform,
    preprocess_transform,
    output_dir,
    pth_name,
):
    # 加载图片
    img = mpimg.imread(image_path)
    if img.ndim == 2:  # 如果是灰度图，转为三通道
        img = np.stack([img, img, img], axis=-1)
    original_img = cv2.cvtColor(
        img, cv2.COLOR_RGB2BGR
    )  # 用于显示和叠加，保持原始 BGR 格式

    # 预处理图片用于模型输入
    input_tensor = preprocess_transform(original_img).unsqueeze(0).to(DEVICE)

    # 获取 Grad-CAM
    cam_extractor = GradCAM(model, target_layer)
    cam = cam_extractor(input_tensor)

    # 调整 Grad-CAM 大小到原图尺寸
    cam_resized = cv2.resize(
        cam,
        (original_img.shape[1], original_img.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    # 生成热力图
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

    # 叠加热力图到原图 (使用透明度)
    # alpha 是热力图的透明度，beta 是原图的透明度，gamma 是亮度偏移
    overlay_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    # 获取模型预测结果
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class_idx = torch.argmax(probabilities).item()

        # 假设类别映射
        class_names = {0: "Abiotic", 1: "Biotic"}
        predicted_class_name = class_names.get(
            predicted_class_idx, f"Class {predicted_class_idx}"
        )
        confidence = probabilities[predicted_class_idx].item() * 100

    # 获取原始标签
    df = pd.read_csv(CSV_PATH)
    rel_path_str = str(image_path.relative_to(ROOT_DIR)).replace(
        "\\", "/"
    )  # 统一路径分隔符
    original_label_row = df[df["rel_path"] == rel_path_str]
    original_label_name = "Unknown"
    if not original_label_row.empty:
        original_label_name = class_names.get(
            original_label_row.iloc[0]["label"],
            f"Class {original_label_row.iloc[0]['label']}",
        )

    # 拼接图片，上方是原始图，下方是叠加图
    # 在图片上方添加文字信息
    text_info = (
        f"GT: {original_label_name} | Pred: {predicted_class_name} ({confidence:.2f}%)"
    )

    # 将 BGR 转换回 RGB 以便 matplotlib 显示
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    overlay_img_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(original_img_rgb)
    axes[0].set_title("Original Image", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(overlay_img_rgb)
    axes[1].set_title(f"Grad-CAM ({text_info})", fontsize=10)
    axes[1].axis("off")

    plt.suptitle(
        f"Attention Map for: {image_path.name}", fontsize=12, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局，为 suptitle 留出空间

    # 保存结果
    image_filename = image_path.stem  # 获取不带扩展名的文件名
    output_filename = f"{image_filename}_{pth_name}.png"
    output_filepath = output_dir / output_filename
    plt.savefig(output_filepath, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)  # 关闭图形，释放内存

    print(f"✅ Grad-CAM 图像已保存至: {output_filepath}")


# ========================== 4. 主执行逻辑 ==========================

if __name__ == "__main__":
    print(f"--- 🚀 启动 Grad-CAM 可视化任务 (使用设备: {DEVICE}) ---")

    # 1. 加载模型
    model = get_plant_model(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 确定目标层 (通常是最后一个卷积层)
    # 假设你的模型是 ResNet 或类似的，通常最后一个 Conv 层在 model.layer4 或 model.features[-1]
    # 你可能需要根据实际模型结构调整这里。这里以一个假设的常见结构为例。
    # 如果是 ResNet, 通常是 model.layer4[-1].conv3 或 model.avgpool 之前的最后一个卷积层
    # 如果是 VGG, 可能是 model.features[-3] 或 model.features[-1] (最后一个 Conv 层)

    # 假设模型的结构是 model.features 或 model.classifier，你需要找到最后一个卷积层
    # 我这里用一个通用的方式尝试找到最后一个卷积层。
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
            print(f"🎯 自动识别到目标 Grad-CAM 层: {name}")

    if target_layer is None:
        print("⚠️ 未找到 Conv2d 层作为 Grad-CAM 目标层。请手动指定 `target_layer`。")
        # 示例：如果模型结构是 ResNet，你可能需要这样指定：
        # target_layer = model.layer4[-1].conv3
        # 或者其他自定义模型的最后一个 Conv 层
        sys.exit("无法继续，请检查模型结构并指定目标层。")

    # 2. 获取预处理函数 (训练时用的 transform，只需推理部分)
    _, test_transform = get_transforms()

    # Pytorch Grad-CAM 库通常要求输入是 PIL Image 或 HWC numpy array
    # 我们的 get_transforms 返回的是针对 PIL Image 的，因此这里进行适配
    # 创建一个用于加载和预处理 Grad-CAM 输入的 transform
    # 这里我们只用 Compose，因为 cv2.imread 已经读入 numpy array
    grad_cam_preprocess_transform = transforms.Compose(
        [
            transforms.ToPILImage(),  # 将 cv2 读取的 numpy array 转换为 PIL Image
            transforms.Resize((224, 224)),  # 模型输入尺寸
            transforms.ToTensor(),  # 转为 Tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # 标准化
        ]
    )

    # 3. 提取 PTH 文件名，用于图片命名
    pth_filename_stem = MODEL_PATH.stem

    # 4. 遍历指定图片进行可视化
    print(
        f"\n--- 📈 正在生成 Grad-CAM 可视化结果 (共 {len(IMAGE_RELATIVE_PATHS_TO_ANALYZE)} 张图) ---"
    )
    for rel_path in tqdm(IMAGE_RELATIVE_PATHS_TO_ANALYZE, desc="处理图片"):
        full_image_path = ROOT_DIR / rel_path
        if not full_image_path.exists():
            print(f"❌ 图片未找到: {full_image_path}，跳过。")
            continue

        visualize_grad_cam(
            model,
            target_layer,
            full_image_path,
            test_transform,
            grad_cam_preprocess_transform,
            OUTPUT_DIR,
            pth_filename_stem,
        )

    print("\n--- ✅ Grad-CAM 可视化任务完成！ ---")
