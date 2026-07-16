from ultralytics import YOLO
import torch

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# Device selection: prefer CUDA, then MPS, then CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

base_model = "yolov8n.pt"
project="temp/checkered_board"

model = YOLO(base_model)

model.train(
    data="data.yaml",
    epochs=15,
    imgsz=640,
    batch=4,
    project=project,
    name="run1",
    device=device,
    save=True,
    val=True
)
