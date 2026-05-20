import cv2
import time
import os
from ultralytics import YOLO
import torch
from collections import deque

WINDOW_SIZE = 30
SAVE_ROOT = "person_windows1"
os.makedirs(SAVE_ROOT, exist_ok=True)

model = YOLO(r"C:\Users\DeepL_10\Desktop\pytorch2\yolo11x.pt")
cap = cv2.VideoCapture(r"C:\Users\DeepL_10\Desktop\歩行データまとめ\渡らない3.mp4")

frame_times = []
frame_count = 0

# 「歩行者が存在したフレームのみ」保存するバッファ
crop_buffer = deque(maxlen=WINDOW_SIZE)

window_index = 1

while cap.isOpened():
    torch.cuda.synchronize()
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, imgsz=640, conf=0.6, verbose=False)
    boxes = results[0].boxes

    frame_crops = []

    if boxes is not None:
        for box, cls in zip(boxes.xyxy, boxes.cls):
            if int(cls) == 0:  # person
                x1, y1, x2, y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]
                frame_crops.append(crop)

    # 歩行者が検出されたフレームのみ追加
    if len(frame_crops) > 0:
        crop_buffer.append(frame_crops)
    else:
        # 連続性が途切れたらリセット（超重要）
        crop_buffer.clear()

    # 30フレーム連続達成したら保存
 
    if len(crop_buffer) == WINDOW_SIZE:
        folder_name = f"{SAVE_ROOT}/window_{window_index:04d}"
        os.makedirs(folder_name, exist_ok=True)

        img_index = 0
        for f_idx, crops_in_frame in enumerate(crop_buffer):
            for crop in crops_in_frame:
                save_path = f"{folder_name}/f{f_idx:02d}_p{img_index:03d}.jpg"
                cv2.imwrite(save_path, crop)
                img_index += 1

        print(f"Saved: {folder_name}")

        window_index += 1
        crop_buffer.popleft()  # ← スライディングウィンドウ維持

    # 表示
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Detection", annotated_frame)

    torch.cuda.synchronize()
    frame_time = time.time() - start_time
    frame_times.append(frame_time)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()