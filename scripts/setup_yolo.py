from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Using YOLOv8 nano model for efficiency

# Run a quick test
model.predict("https://ultralytics.com/images/bus.jpg", save=True)

print("YOLO setup completed successfully!")
