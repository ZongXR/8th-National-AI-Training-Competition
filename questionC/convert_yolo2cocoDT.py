import json
import time
import os
import cv2

# 文件信息标签
with open(r"ImgInfo_label.json", "r") as rf:
    dataset = json.load(rf)

# yolo 标签文件夹
label_dir = r"labelData2/labels"

cocoDT = []
for imgInfo in dataset["images"]:
    img_name = imgInfo["file_name"]
    img_id = imgInfo["id"]
    W = imgInfo["width"]
    H = imgInfo["height"]

    txtFile = f'{img_name[:img_name.rfind(".")]}.txt'
    if not os.path.exists(os.path.join(label_dir, txtFile)):
        continue
    
    with open(os.path.join(label_dir, txtFile), 'r') as rf:
        labelList = rf.readlines()
        for label in labelList:
            label = label.strip().split()
            x = float(label[1])
            y = float(label[2])
            w = float(label[3])
            h = float(label[4])

            # convert x,y,w,h to x1,y1,x2,y2
            x1 = int((x - w / 2) * W )
            y1 = int((y - h / 2) * H )
            x2 = int((x + w / 2) * W )
            y2 = int((y + h / 2) * H )

            # 标签序号从图像 id 的 10000 倍 开始计算 。
            ann_id_cnt = img_id * 10000

            cls_id = int(label[0])   
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            cocoDT.append({
                'bbox': [x1, y1, width, height],
                'category_id': cls_id,
                'image_id': img_id,
                "score": 0.9
            })
            ann_id_cnt += 1

# 输出标注的最终结果文件夹
with open(r"label.json", "w") as wf:
    json.dump(cocoDT, wf)
