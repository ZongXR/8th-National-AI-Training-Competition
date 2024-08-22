# -*- coding: utf-8 -*-
import json
import time
import os
import cv2
from ultralytics import YOLO


model = YOLO("/path/best.pt")  # build from YAML and transfer weights

success = model.export(format="onnx", imgsz=640)

