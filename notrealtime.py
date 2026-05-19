import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image  
from playsound import playsound
import threading
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import time



#学習済みCNNモデルをロード

cnn_model = models.mobilenet_v2(weights=None)
cnn_model.classifier[1] = nn.Linear(cnn_model.last_channel, 2)
cnn_model.load_state_dict(torch.load("pedestrian_classifier.pth", map_location="cuda"))
cnn_model.eval()

device = torch.device("cuda")
cnn_model = cnn_model.to(device)


#  画像前処理

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# YOLOモデル

yolo_model = YOLO(r"C:\Users\DeepL_10\Desktop\pytorch\yolo11x.pt")


#  入力動画

video_path = r"C:\Users\DeepL_10\Desktop\渡らない7.mp4"
cap = cv2.VideoCapture(video_path)

# 入力動画
video_path = r"C:\Users\DeepL_10\Desktop\渡らない7.mp4"
cap = cv2.VideoCapture(video_path)

# 最初のフレームを読み込む
ret, first_frame = cap.read()
if not ret:
    print("動画を読み込めませんでした")
    exit()

# ここで正しいサイズを取得する
height, width = first_frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)

# 出力動画の設定
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output17.mp4", fourcc, fps, (width, height))

# 最初のフレームを戻す
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


# 出力動画の設定
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output18.mp4", fourcc, fps, (width, height))

# ===== 注意・警告の表示管理 =====
alert_text = ""
alert_color = (0, 0, 0)
alert_start_time = 0
alert_duration = 3  # 3秒表示
#  動画フレームごとの処理

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    crossing_prob = 0 

    # YOLOで人物検出
    results = yolo_model.predict(frame, imgsz=640, conf=0.6, verbose=False)

    # 各人物をCNNで推論
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls != 0:  # person以外はスキップ
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                continue

            # OpenCV → PIL
            person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            person_pil = Image.fromarray(person_img_rgb)

            # CNN推論
            person_tensor = transform(person_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = cnn_model(person_tensor)
                probs = torch.softmax(outputs, dim=1)
                crossing_prob = probs[0][1].item() * 100

            # 描画
            label = f"{crossing_prob:.1f}%"
            color = (0, 0, 255) if crossing_prob > 50 else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    
        

        

    # バー全体の位置とサイズ
    bar_x, bar_y = width - 320, 80
    bar_width, bar_height = 160, 880

    

    # 外枠（白）
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)

    # 確率に応じた棒の高さ
    filled_height = int((crossing_prob / 100) * bar_height)

    # --- 色のグラデーション（0→50→100：緑→黄→赤） ---
    if crossing_prob <= 50:
        # 緑→黄（0～50）
        ratio = crossing_prob / 50
        r = int(255 * ratio)       # 赤が増える
        g = 255                    # 緑は最大
        b = 0                      # 青なし
    else:
        # 黄→赤（50～100）
        ratio = (crossing_prob - 50) / 50
        r = 255                    # 赤は最大
        g = int(255 * (1 - ratio)) # 緑が減る
        b = 0
    color = (b, g, r)  # OpenCVはBGR順

    # 棒の塗りつぶし（下から上へ）
    cv2.rectangle(frame,
                  (bar_x, bar_y + bar_height - filled_height),
                  (bar_x + bar_width, bar_y + bar_height),
                  color, -1)

    # 白い枠で再度囲む（視認性アップ）
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)

    # テキスト表示（中央下）
    text = f"{crossing_prob:.1f}%"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = bar_x + bar_width // 2 - text_size[0] // 2
    text_y = bar_y + bar_height + 45
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    
    # ======== 警告表示と音 ========
    current_time = time.time()

    if crossing_prob > 80:
        if alert_text != " 危険！！ ":
            alert_text = " 危険！！ "
            alert_color = (255, 0, 0)
            alert_start_time = current_time
    elif crossing_prob > 50:
        if alert_text not in [" 危険！！ ", " 注意！ "]:
            alert_text = " 注意！ "
            alert_color = (255, 255, 0)
            alert_start_time = current_time

    # === 表示（3秒間固定） ===
    if alert_text and (current_time - alert_start_time < alert_duration):
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        # フォントサイズを条件で変える
        if alert_text == " 危険！！ ":
            font = ImageFont.truetype("C:/Windows/Fonts/msgothic.ttc", 180)
        else:
            font = ImageFont.truetype("C:/Windows/Fonts/msgothic.ttc", 120)

        bbox = draw.textbbox((0, 0), alert_text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((width - w) / 2, 80), alert_text, fill=alert_color, font=font)

        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    else:
        alert_text = ""  # 時間経過でリセット

    out.write(frame)

cap.release()
out.release()
print(" output.mp4 に保存されました！")
