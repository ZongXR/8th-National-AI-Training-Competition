import json
import typing
import numpy as np
import refile
import yaml
from snapdata.io import load_sds_iter
import os
from snapdata2 import sds_open


def xyxy2xywhn1(x, w=640, h=640):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[0] = ((x[0] + x[2]) / 2) / w  # x center
    y[1] = ((x[1] + x[3]) / 2) / h  # y center
    y[2] = (x[2] - x[0]) / w  # width
    y[3] = (x[3] - x[1]) / h  # height
    return y


def convert_sds_to_yolo(
    data_sds: typing.Union[list, dict], output_dir: str
):
    """
    将 sds 格式的文件转换为 yolo 的格式，支持检测任务。
    train_sds: 数据集sds文件的路径，可以是一个列表，也可以是一个字典，key 为数据集的名称，value 为 sds 文件的路径。
    output_dir: 输出的文件夹根路径。
    使用样例：convert_sds_to_yolo（s3://snapdet-benchmark/sds_data/6812c58214357abe44278fa3edb9f31c/data.sds, "outdir")
    """

    if isinstance(data_sds, dict):
        data_sds = list(data_sds.values())
    print(data_sds)
    image_folder_path = refile.smart_path_join(output_dir, "images")
    label_folder_path = refile.smart_path_join(output_dir, "labels")

    refile.smart_makedirs(output_dir, exist_ok=True)
    refile.smart_makedirs(image_folder_path, exist_ok=True)
    refile.smart_makedirs(label_folder_path, exist_ok=True)

    print(os.getenv("OSS_ENDPOINT"))
    print(os.getenv("AWS_SECRET_ACCESS_KEY"))
    print(os.getenv("AWS_ACCESS_KEY_ID"))
    
    images = []
    category_name_to_category_id = {
        "Pedestrian": 0, "Cyclist": 1,
        "Car": 2, "Truck": 3,
        "Tram": 4, "Tricycle": 5,
    }

    for sds_path in data_sds:
        # load sds
        with sds_open(sds_path) as sds_iter:
        #sds_iter = load_sds_iter(sds_path)
            print(type(sds_iter))
        # convert sds to YOLO format
            for line in sds_iter:
                shape = (line["image_height"], line["image_width"])
                image_filename = line['extra']['original']['filename']
                image = {
                    "width": shape[1],
                    "height": shape[0],
                    "file_name": image_filename,
                }
                images.append(image)

                image_path = refile.smart_path_join(
                    image_folder_path, image_filename
                )

                labels, categories, raw_lb = [], [], []
                print(line["url"])
                print(image_path)
                refile.smart_copy(line["url"], image_path)

                for b in line["boxes"]:
                    if b["type"] == "detected_box" or b["type"] == "ignored_box":
                        continue

                    class_name = b["class_name"]
                    if class_name not in category_name_to_category_id:
                        category_id = len(category_name_to_category_id)
                        category_name_to_category_id[class_name] = category_id
                    category_id = category_name_to_category_id[class_name]
                    lb = xyxy2xywhn1(
                        np.array(
                            [
                                max(b["x"], 0),
                                max(b["y"], 0),
                                min(b["x"] + b["w"], shape[1]),
                                min(b["y"] + b["h"], shape[0]),
                            ],
                            dtype=np.float32,
                        ),
                        w=shape[1],
                        h=shape[0],
                    )
                    categories.append(category_id)
                    labels.append(lb)
                    raw_lb.append([category_id, *lb])

                label_path = refile.smart_path_join(
                    label_folder_path, image_filename.replace("jpg", "txt")
                )
                with refile.smart_open(label_path, "w") as f:
                    for class_lb in raw_lb:
                        f.write(" ".join([str(i) for i in class_lb]) + "\n")

# 示例
# 训练平台数据集 sds地址 
# 图像及标签保存文件夹
data_sds = ["s3://system-testqaqmxdrb-dataset/dataset-66b10692c36bf9a154ed756d/revision-66b33868c36bf9a154ed7a59/entity.sds"]
output_dir = "labelData"

convert_sds_to_yolo(data_sds, output_dir)

