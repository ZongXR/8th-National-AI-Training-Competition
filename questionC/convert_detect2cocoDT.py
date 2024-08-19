import json
import cv2
from ultralytics import YOLO
import os

with open(r"/home/aiservice/workspace/questionC/ImgInfo_test.json", "r") as rf:
    dataset = json.load(rf)

image_dir = r"yoloTrain/dataset/images/val0"


nms_threshold = 0.3
confidence_threshold = 0.4

model = YOLO(r"/home/aiservice/workspace/questionC/yoloTrain/runs/detect/train3/weights/last.pt")

cocoDT = []
for imgInfo in dataset["images"]:
    img_name = imgInfo["file_name"]
    img_id = imgInfo["id"]
    W = imgInfo["width"]
    H = imgInfo["height"]

    if not os.path.exists(os.path.join(image_dir, img_name)):
        continue

    # 进行检测
    img = cv2.imread(os.path.join(image_dir, img_name))
    results = model.predict(source=img, conf=confidence_threshold, iou=nms_threshold)

    for box in results[0].boxes:
        ccls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
        cocoDT.append({
            'bbox': [x1, y1, x2 - x1, y2 - y1],
            'category_id': ccls,
            'image_id': img_id,
            "score": conf
        })


with open(r"detect.json", "w") as wf:
    json.dump(cocoDT, wf)