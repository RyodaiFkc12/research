import os
import numpy as np
import pandas as pd

DATASET_DIR = "LSTMdataset"

X = []
y = []

for label_name in ["渡る", "渡らない"]:
    label = 1 if label_name == "渡る" else 0
    folder = os.path.join(DATASET_DIR, label_name)

    for seq in os.listdir(folder):
        csv_path = os.path.join(folder, seq, "features.csv")
        df = pd.read_csv(csv_path)

        # 30フレームチェック
        if len(df) != 30:
            continue

        seq_data = df.drop("frame", axis=1).values  # (30,1280)
        X.append(seq_data)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)  # (N,30,1280)
print("y shape:", y.shape)

print("渡る:", np.sum(y == 1))
print("渡らない:", np.sum(y == 0))