# -*- coding: utf-8 -*-
import os
import random
from datetime import datetime
from tqdm import tqdm
from ultralytics import YOLO


if not os.path.exists("./data/images/val"):
    os.mkdir("./data/images/val")
else:
    for file in os.listdir("./data/images/val"):
        os.rename(os.path.join("./data/images/val", file), os.path.join("./data/images/train", file))

if not os.path.exists("./data/labels/val"):
    os.mkdir("./data/labels/val")
else:
    for file in os.listdir("./data/labels/val"):
        os.rename(os.path.join("./data/labels/val", file), os.path.join("./data/labels/train", file))


all_labels = os.listdir("./data/labels/train")
val_size = 0.2
random.seed(42)
val_labels = random.sample(all_labels, int(val_size * len(all_labels)))
train_labels = [x for x in all_labels if x not in val_labels]
print(f"训练集数量: {len(train_labels)}")
print(f"验证集数量: {len(val_labels)}")
for file in tqdm(val_labels):
    os.rename(os.path.join("./data/images/train", file.split(".")[0] + ".jpg"), os.path.join("./data/images/val", file.split(".")[0] + ".jpg"))
    os.rename(os.path.join("./data/labels/train", file), os.path.join("./data/labels/val", file))

model = YOLO("./yolov8s.pt")
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results = model.train(
    task="detect",
    model="./yolov8s.pt",
    data="./datasets.yaml",
    epochs=100,
    batch=8,
    imgsz=1920,
    patience=50,
    save=True,
    save_period=5,
    device=0,
    workers=10,
    seed=42,
    name=start_time,
    pretrained=True,
    optimizer="auto",
    cos_lr=False,
    plots=True,
    deterministic=True,
    freeze=None,
    save_json=True,
    cache=False,
    mixup=0.5
)
