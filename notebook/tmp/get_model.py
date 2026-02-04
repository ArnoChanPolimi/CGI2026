import torch
import torchvision.models as models

print("正在检查环境...")
print(f"当前 Python 版本: {torch.__version__}")

try:
    # 尝试下载预训练的 MobileNetV2 权重
    # weights='DEFAULT' 会自动从官网抓取模型
    model = models.mobilenet_v2(weights="DEFAULT")
    print("\n--- 成功！ ---")
    print("MobileNetV2 模型已成功下载并加载到 cgi_leaf 环境中。")
    print(f"模型最后一层结构: {model.classifier[1]}")
except Exception as e:
    print(f"\n--- 失败 ---")
    print(f"模型下载出错了，原因: {e}")
