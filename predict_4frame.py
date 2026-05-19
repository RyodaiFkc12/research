import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import torch.nn.functional as F


#  設定
image_dir = r"C:\Users\DeepL_10\Desktop\test5"  # 画像フォルダ
class_names = ["wataranai", "wataru"]  # クラス名
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  モデル定義（4フレーム対応MobileNetV2）
model = models.mobilenet_v2(weights=None)

# 3ch × 4フレーム = 12チャンネル入力
model.features[0][0] = nn.Conv2d(12, 32, kernel_size=3, stride=2, padding=1, bias=False)

# 出力層（2クラス分類）
model.classifier[1] = nn.Linear(model.last_channel, 2)

# 学習済みモデルの重みを読み込む
model.load_state_dict(torch.load("pedestrian_4frame_cnn.pth", map_location=device))
model.eval().to(device)



#  前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

#  4フレームを1サンプルとして推論
# フォルダ内の画像をソート（image0001.jpg → image0002.jpg → ...）
image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])

# 4枚ずつ処理
for i in range(0, len(image_files) - 3, 4):
    imgs = []
    frame_names = image_files[i:i+4]  # 4枚取得

    # 各画像を前処理してテンソル化
    for fname in frame_names:
        img_path = os.path.join(image_dir, fname)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img)
        imgs.append(img_tensor)

    # 4枚分をチャンネル方向に結合 → [1, 12, 224, 224]
    input_tensor = torch.cat(imgs, dim=0).unsqueeze(0).to(device)

    # 推論
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]
        confidence = probs[0][predicted.item()].item() * 100

    print(f"{frame_names} → 判定結果：この歩行者は「{label}」と推定されました ")
    print(f" {label} である確率は{confidence:.2f}% です。")


