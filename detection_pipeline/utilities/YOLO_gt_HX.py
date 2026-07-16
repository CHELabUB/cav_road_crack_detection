
# This function is written to use YOLO generate some ground truth verification
# author: Haosong Xiao
# requires the model file best.pt in the same folder
# one can download trained model from: https://buffalo.box.com/s/6xe3urkqnz4p5l5uggyf5wd5oyo2t6ix

# native imports
import os
# need to pip install ultralytics for YOLO model loading and inference
from ultralytics import YOLO


def Yolo_gt_HX(img, confidence_threshold=0.1, model_path=None):

    if model_path is None:
        print("please check the model path, it's missing, check from UB box")
        return []

    model = YOLO(model_path)
    results = model.predict(source=img, save=False, verbose=False)

    bounding_boxes = []
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                confidence = box.conf[0].item()
                print('current confidence', confidence)
                if confidence >= confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bounding_boxes.append([int(x1), int(y1), int(x2), int(y2)])

    if not bounding_boxes:
        bounding_boxes.append([-1, -1, -1, -1])

    return bounding_boxes
