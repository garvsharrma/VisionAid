from ultralytics import YOLO
import cv2
import sys

# Load the YOLO model
model = YOLO("yolov8n.pt")  # Using YOLOv8 nano model

# Get the image path from command-line arguments
if len(sys.argv) < 2:
    print("Usage: python detect_image.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

# Run inference on the image
results = model.predict(image_path, save=True)

# Display the result
for result in results:
    image_with_detections = result.plot()
    cv2.imshow("YOLO Detection", image_with_detections)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
