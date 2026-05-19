import os
import glob
import numpy as np
import pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image

# 設定 
IMAGE_FOLDER = r"C:\Users\DeepL_10\Desktop\pytorch\試し30フレーム\window_0056"      # 30枚画像フォルダ
OUTPUT_CSV = "features.csv"
IMG_SIZE = 224

# MobileNetV2 読み込み
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(pretrained=True)
model.classifier = torch.nn.Identity()   # 1280次元特徴量
model.to(device)
model.eval()

# 前処理 
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 画像取得
image_paths = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")))

assert len(image_paths) == 30, "30フレーム必要です"

features_list = []

# 特徴量抽出
for idx, img_path in enumerate(image_paths):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = model(img)          # (1,1280)
        feature = feature.cpu().numpy().flatten()

    row = [idx] + feature.tolist()   # 先頭にフレーム番号追加
    features_list.append(row)

# CSV保存 
columns = ["frame"] + [f"f{i}" for i in range(1280)]
df = pd.DataFrame(features_list, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print("CSV保存完了:", OUTPUT_CSV)