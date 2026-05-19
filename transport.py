import os
import random
import shutil

# 元の画像フォルダ
src_dir = r"C:\Users\DeepL_10\Desktop\dataset\train\wataru"

# 移動先フォルダ
dst_dir = r"C:\Users\DeepL_10\Desktop\test6渡る"

os.makedirs(dst_dir, exist_ok=True)

# 対象ファイル
exts = (".jpg", ".jpeg", ".png", ".bmp")
files = [f for f in os.listdir(src_dir) if f.lower().endswith(exts)]

# 100枚選ぶ
selected = random.sample(files, min(100, len(files)))

# ファイル移動
for f in selected:
    shutil.move(os.path.join(src_dir, f), os.path.join(dst_dir, f))

print(f"移動完了: {len(selected)} 枚")
