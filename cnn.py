import torch
import torch.nn as nn
from torchvision import models

# MobileNetV2ベースのモデル（学習時と同じ構造にする）
cnn_model = models.mobilenet_v2(pretrained=False)
cnn_model.classifier[1] = nn.Linear(cnn_model.last_channel, 2)  # 2クラス分類用に修正

# 学習済みパラメータをロード
cnn_model.load_state_dict(torch.load("pedestrian_4frame_cnn.pth", map_location="cpu"))

# 評価モードに設定
cnn_model.eval()

print(" 学習済みモデルを正しくロードしました")

