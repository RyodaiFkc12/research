import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import deque

# 学習済みCNNモデル（4フレーム入力版）をロード

class FourFrameMobileNet(nn.Module):
    def __init__(self, base_model, num_classes=2):
        super().__init__()
        self.features = base_model.features
        # 入力チャンネル数を 3 → 12 に変更（4枚のRGB画像）
        self.features[0][0] = nn.Conv2d(12, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(base_model.last_channel, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


# モデルを構築してロード
base_model = torch.hub.load("pytorch/vision", "mobilenet_v2", pretrained=False)
cnn_model = FourFrameMobileNet(base_model, num_classes=2)
cnn_model.load_state_dict(torch.load("pedestrian_4frame_cnn.pth", map_location="cuda"))
cnn_model.eval()

device = torch.device("cuda")
cnn_model = cnn_model.to(device)


# 前処理

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# YOLOモデル

yolo_model = YOLO(r"C:\Users\DeepL_10\Desktop\pytorch\yolo11x.pt")


#  動画設定
video_path = r"C:\Users\DeepL_10\Desktop\渡る.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_4frame2.mp4", fourcc, fps, (width, height))

# 直近4フレームを保存するキュー
frame_buffer = deque(maxlen=4)


#  推論ループ
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_buffer.append(frame)
    crossing_prob = 0

    # フレームが4枚たまったら推論
    if len(frame_buffer) == 4:
        results = yolo_model.predict(frame, imgsz=640, conf=0.6, verbose=False)

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls != 0:  # person以外はスキップ
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_imgs = []

                # 各フレームから同じ領域を切り出す
                for f in frame_buffer:
                    crop = f[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_pil = Image.fromarray(crop_rgb)
                    person_imgs.append(transform(crop_pil))

                if len(person_imgs) == 4:
                    # 4フレームをチャンネル方向に結合 → (12, 224, 224)
                    input_tensor = torch.cat(person_imgs, dim=0).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = cnn_model(input_tensor)
                        probs = torch.softmax(outputs, dim=1)
                        crossing_prob = probs[0][1].item() * 100

                    label = f"{crossing_prob:.1f}%"
                    color = (0, 0, 255) if crossing_prob > 50 else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  
    # 右上に確率バーを描画

    bar_x, bar_y = width - 150, 50
    bar_width, bar_height = 30, 200

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (180, 180, 180), 2)

    filled_height = int((crossing_prob / 100) * bar_height)
    color = (0, 255, 0) if crossing_prob <= 50 else (0, 0, 255)

    cv2.rectangle(frame,
                  (bar_x, bar_y + bar_height - filled_height),
                  (bar_x + bar_width, bar_y + bar_height),
                  color, -1)

    cv2.putText(frame, f"{crossing_prob:.1f}%", (bar_x - 10, bar_y + bar_height + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out.write(frame)

cap.release()
out.release()
print(" output_4frame.mp4 に保存されました。")
