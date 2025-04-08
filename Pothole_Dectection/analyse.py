from ultralytics import YOLO

# Load your YOLO model
model_path = "best_1.pt"  # Update this path if necessary
model = YOLO(model_path)

# Extract and display model information
print("Number of Classes:", len(model.names))
print("Class Names:", model.names)
