from ultralytics import YOLO

model_name = "path to your trained model"
path_to_test_images = "path/to/test/images/"
output_filename = "define output filename here.txt"
output_path = "where/to/save/predicted/images/"

# example
# model_name = "temp/best.pt"
# output_filename = "temp/output_predict.txt"
# path_to_test_images = "temp/predict"
# output_path = "temp/predict"

# performing prediction on the test images
model = YOLO(model_name)

results = model.predict(source=path_to_test_images, save=True, project=output_path)

# finding the bounding box coordinates of the detected checkered boards
for r in results:
    image_filename = r.path.split("/")[-1]
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        with open(output_filename, "a") as f:
            f.write(f"Image Name: {image_filename}, u: {x_center}, v: {y_center}\n")
