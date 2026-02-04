import torch
import torchvision
import cv2
import matplotlib

print("--- 检查结果 ---")
print(f"PyTorch 版本: {torch.__version__}")
print(f"Torchvision 版本: {torchvision.__version__}")
print(f"OpenCV (cv2) 是否可用: {'是' if cv2.__version__ else '否'}")
print(f"能否检测到显卡 (CUDA): {torch.cuda.is_available()}")
print("---------------")
