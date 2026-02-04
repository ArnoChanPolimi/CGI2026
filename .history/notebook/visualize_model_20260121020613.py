import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# --- 1. 核心纠正：直接从根目录开始找 ---
# 你在 CGI_project 下运行，所以直接写 data/... 就行了，不要加 ..
img_path = os.path.join(
    "data", "train", "abiotic", "Nutrition_Deficiency_Nutrition Deficiency_003.jpg"
)

# 输出到同级的 tmp
output_folder = "tmp"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "final_comparison.png")

# --- 2. 验证路径并读取 ---
if not os.path.exists(img_path):
    print(f"❌ 我还是瞎了！请看这个路径对不对：")
    print(f"绝对路径：{os.path.abspath(img_path)}")
    # 打印当前文件夹下的内容，帮你排查
    print(f"当前目录下有的文件夹：{os.listdir('.')}")
    import sys

    sys.exit(1)

img = Image.open(img_path).convert("RGB")
w, h = img.size
print(f"✅ 终于抓到图了！原始尺寸: {w}x{h}")

# --- 3. 三种逻辑对比（重点：看看 300x3000 怎么变 224） ---
# 逻辑 A：你现在的 (Resize + CenterCrop) -> 会剪掉两头
logic_1 = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])

# 逻辑 B：我推荐的 (RandomResizedCrop) -> 高清采样
logic_2 = transforms.RandomResizedCrop(224, scale=(0.5, 1.0))


# 逻辑 C：保全但模糊 (Pad + Resize) -> 缩成牙签
def pad_resize(image):
    max_s = max(image.size)
    p = transforms.Compose(
        [
            transforms.Pad(
                ((max_s - image.size[0]) // 2, (max_s - image.size[1]) // 2), fill=0
            ),
            transforms.Resize((224, 224)),
        ]
    )
    return p(image)


# --- 4. 绘图对比 ---
plt.figure(figsize=(20, 6))
plt.subplot(1, 4, 1)
plt.imshow(img)
plt.title(f"Original\n({w}x{h})")
plt.axis("off")
plt.subplot(1, 4, 2)
plt.imshow(logic_1(img))
plt.title("Your Current Logic\n(Info Lost!)")
plt.axis("off")
plt.subplot(1, 4, 3)
plt.imshow(logic_2(img))
plt.title("Proposed: RandomCrop\n(High Res)")
plt.axis("off")
plt.subplot(1, 4, 4)
plt.imshow(pad_resize(img))
plt.title("Pad & Resize\n(Too Thin)")
plt.axis("off")

plt.tight_layout()
plt.savefig(output_path)
print(f"🚀 结果已生成：{output_path}")
