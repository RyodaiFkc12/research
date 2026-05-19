import torch
import torch.nn as nn
from torchvision import models

# MobileNetV2ベースのモデル
cnn_model = models.mobilenet_v2(pretrained=False)

# 4枚入力（3ch×4=12ch）に変更
cnn_model.features[0][0] = nn.Conv2d(12, 32, kernel_size=3, stride=2, padding=1, bias=False)

# 出力層（2クラス分類）
cnn_model.classifier[1] = nn.Linear(cnn_model.last_channel, 2)

# 学習済み重みを読み込む
cnn_model.load_state_dict(torch.load("pedestrian_4frame_cnn.pth", map_location="cuda"))

# 評価モード
cnn_model.eval()

