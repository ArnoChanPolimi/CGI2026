import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# --- 1. 严格路径设置 ---
# 你指定的图片路径
img_relative_path = os.path.join(
    "data", "train", "abiotic", "Nutrition_Deficiency_Nutrition Deficiency_003.jpg"
)
output_folder = "tmp"
os.makedirs(output_folder, exist_ok=True)
comparison_output_path = os.path.join(output_folder, "final_image_comparison.jpg")

# --- 2. 最终处理逻辑定义 (包含随机旋转) ---
# 这是模型最终会“吃”进去的预处理流程
final_model_input_transform = transforms.Compose(
    [
        transforms.Resize(256),  # 短边缩放至256，长边等比，利用白边缓冲
        transforms.CenterCrop(224),  # 从256中心裁剪224，物理裁掉外部空白
        transforms.RandomChoice(
            [  # 随机选择一个角度，模拟模型可能看到的四种方向
                transforms.RandomRotation((0, 0)),
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((180, 180)),
                transforms.RandomRotation((270, 270)),
            ]
        ),
    ]
)

# --- 3. 执行对比并保存 ---
if not os.path.exists(img_relative_path):
    print(f"❌ 找不到文件！请检查路径：{os.path.abspath(img_relative_path)}")
else:
    original_img = Image.open(img_relative_path).convert("RGB")
    print(f"✅ 抓到图了！原始尺寸: {original_img.size}")

    # 生成处理后的图像（这里随机取一个旋转角度）
    processed_img = final_model_input_transform(original_img)

    # --- 4. 可视化对比 ---
    plt.figure(figsize=(12, 6))  # 调整图幅大小，让两张图都能看清

    plt.subplot(1, 2, 1)  # 左边是原图
    plt.imshow(original_img)
    plt.title(
        f"Original Image\n{original_img.size[0]}x{original_img.size[1]}", fontsize=14
    )
    plt.axis("off")

    plt.subplot(1, 2, 2)  # 右边是处理后的图
    plt.imshow(processed_img)
    plt.title(
        f"Processed for Model Input\n{processed_img.size[0]}x{processed_img.size[1]} (with random rotation)",
        fontsize=14,
    )
    plt.axis("off")

    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.savefig(comparison_output_path)
    print(f"🚀 对比图已生成并保存至: {comparison_output_path}")

print("\n--- 任务完成 ---")
