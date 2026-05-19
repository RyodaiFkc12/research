import os

# 画像が入っているフォルダのパス
folder_path = r"C:\Users\DeepL_10\Desktop\dataset\val\wataru"

# 画像拡張子のリスト
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

# カウント
count = 0
for filename in os.listdir(folder_path):
    if filename.lower().endswith(image_extensions):
        count += 1

print(f"画像の枚数: {count} 枚")
