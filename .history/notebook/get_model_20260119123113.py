import torchvision.models as models

try:
    # 尝试加载预训练的 MobileNetV2
    model = models.mobilenet_v2(weights="DEFAULT")
    print("MobileNetV2 结构与权重下载成功！")
except Exception as e:
    print(f"下载失败，原因: {e}")
