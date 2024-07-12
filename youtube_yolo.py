

import cv2
from ultralytics import YOLO
import os
import argparse
from pytubefix import YouTube

# Function to download YouTube video using pytubefix
def download_youtube_video(url, output_path):
    try:
        yt = YouTube(url)
        stream = yt.streams.get_highest_resolution()
        stream.download(output_path=output_path, filename="downloaded_video.mp4")
        print("Video downloaded successfully.")
    except Exception as e:
        print(f"Error downloading video: {e}")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="YOLOv5 Object Detection on YouTube Video")
parser.add_argument("url", type=str, help="URL of the YouTube video to process")
args = parser.parse_args()



# Load the pretrained YOLOv5 model
# model = YOLO(model_path)


# Download the YOLOv9 model if it doesn't exist
model_path = "yolov9c.pt"
# Load the pretrained YOLOv9 model
model = YOLO(model_path)


# Download YouTube video
youtube_url = args.url
download_youtube_video(youtube_url, os.getcwd())

# Open the downloaded video
video_path = "downloaded_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # Convert the frame to the format expected by YOLO
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLO model on the frame
    results = model.predict(frame_rgb)

    # Draw bounding boxes on the frame
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
        class_id = int(box.cls)
        class_name = results[0].names[class_id]
        confidence = box.conf.item()
        if confidence < 0.5:
            continue
        
        # Draw rectangle and label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow("YOLO Video Detection", frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video and close the window
cap.release()
cv2.destroyAllWindows()
