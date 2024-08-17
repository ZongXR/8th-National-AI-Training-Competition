from ultralytics import YOLO

import json
import time
import os
import cv2
from ultralytics import YOLO

# 加载 YOLOv8 模型
model = YOLO("/home/aiservice/workspace/questionC/yoloTrain/runs/detect/train3/weights/last.pt")  # build from YAML and transfer weights


# 读取图像
img_dir = r"/home/aiservice/workspace/test/images"
out_img_dir = r"/home/aiservice/workspace/test/out_img"

for root, dirs, files in os.walk(img_dir):
    for f in files:
        input_path = os.path.join(root, f)
        out_path = os.path.join(out_img_dir, f)

        # img_path = '/home/aiservice/workspace/questionC/yoloTrain/dataset/images/val/HT_VAL_000095_SH_210.jpg'
        img = cv2.imread(input_path)


        nms_threshold = 0.3
        confidence_threshold = 0.4

        # 进行检测
        # results = model(img)
        results = model.predict(source=img, conf=confidence_threshold, iou=nms_threshold)


        # 获取检测框和标签
        boxes = results[0].boxes.xyxy  # xyxy format
        confidences = results[0].boxes.conf
        class_ids = results[0].boxes.cls

        # 加载类别名称（根据需要修改路径）
        class_names = model.names

        # 绘制检测框
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f'{class_names[int(class_id)]}: {confidence:.2f}'
            color = (0, 255, 0)  # 绿色框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 保存可视化结果
        # output_path = 'output_image.jpg'
        cv2.imwrite(out_path, img)

        # print(f"Detection results saved to {out_path}")
