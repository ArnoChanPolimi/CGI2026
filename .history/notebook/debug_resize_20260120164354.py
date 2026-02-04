import os
import sys
from torchvision import transforms
from PIL import Image

# 1. 定死逻辑：等比例缩放(Resize) + 画布补齐(CenterCrop)
# 针对 300x3000 -> 保持比例缩成细线，左右补黑，不压扁不剪断
process_logic = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ]
)


def run_debug():
    # 自动创建 tmp 文件夹
    save_dir = os.path.join(os.getcwd(), "tmp")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 获取输入路径：支持直接拖拽文件到脚本，或者手动输入
    if len(sys.argv) > 1:
        img_path = sys.argv[1].strip('"')  # 处理拖拽带来的双引号
    else:
        print("=" * 30)
        img_path = input("请直接粘贴【长条图】的完整路径并按回车: ").strip('"')
        print("=" * 30)

    if not os.path.exists(img_path):
        print(f"错误：路径不存在 -> {img_path}")
        return

    try:
        # 读取并处理
        img = Image.open(img_path).convert("RGB")
        processed_img = process_logic(img)

        # 保存
        base_name = os.path.basename(img_path)
        save_path = os.path.join(save_dir, f"check_{base_name}")
        processed_img.save(save_path)

        print(f"\n成功！处理后的图已存入: {save_path}")
        print(f"原图比例: {img.size[0]}x{img.size[1]}")
        print("去 tmp 文件夹看吧，长条图现在应该变成了一根【不压扁】的细线。")

    except Exception as e:
        print(f"处理失败，错误原因: {e}")


if __name__ == "__main__":
    run_debug()
