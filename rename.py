import os

# 対象のフォルダパス
folder_path = r"C:\Users\DeepL_10\Desktop\pytorch\runs\detect\predict30\crops\person"

# フォルダ内のすべてのファイルをチェック
for filename in os.listdir(folder_path):
    # 拡張子が.jpgのファイルのみ対象
    if filename.lower().endswith(".jpg"):
        name, ext = os.path.splitext(filename)
        new_name = f"(24){name}{ext}"
        original_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)

        # すでにリネーム後のファイルが存在するかチェック
        if not os.path.exists(new_path):
            os.rename(original_path, new_path)
            print(f"{filename} → {new_name} に変更しました。")
        else:
            print(f"{new_name} は既に存在するためスキップしました。")
