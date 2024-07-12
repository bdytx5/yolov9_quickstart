# hello 

from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO
import os

# Download the YOLOv9 model if it doesn't exist
model_path = "yolov9m.pt"
# Load the pretrained YOLOv9 model
model = YOLO(model_path)

# Download the image from the URL
image_url = "https://di-uploads-pod25.dealerinspire.com/koenigseggflorida/uploads/2019/08/Koenigsegg_TheSquad_3200x2000-UPDATED.jpg"
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))

# Set the confidence and IoU thresholds
confidence_threshold = 0.5
iou_threshold = 0.4

# Predict with the model using the set thresholds
results = model.predict(img, conf=confidence_threshold, iou=iou_threshold)
results[0].show()
