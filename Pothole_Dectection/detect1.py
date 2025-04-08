import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load the YOLO model
model = YOLO("best5.pt")
class_names = model.names

# Open the video file
cap = cv2.VideoCapture("pothole.mp4")

# Set desired frame rate (e.g., process every nth frame)
desired_frame_rate = 30  # Target frames per second
frame_skip = int(cap.get(cv2.CAP_PROP_FPS) / desired_frame_rate)  # Skip frames based on video FPS

while True:
    start_time = time.time()
    ret, img = cap.read()
    if not ret:
        break

    # Skip frames to increase the effective output frame rate
    if frame_skip > 1:
        for _ in range(frame_skip - 1):
            cap.read()

    # Resize the frame for consistent input size
    img = cv2.resize(img, (640, 480))
    h, w = img.shape[:2]

    # Run the model prediction with a higher confidence threshold
    res = model.predict(img, conf=0.5)  # Increased confidence threshold to reduce false positives

    for r in res:
        boxes = r.boxes
        masks = r.masks

        if masks is not None:
            masks = masks.data.cpu()
            for seg, box in zip(masks.cpu().numpy(), boxes):
                seg = cv2.resize(seg, (w, h))
                contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cont in contours:
                    d = int(box.cls)
                    c = class_names[d]
                    x, y, bw, bh = cv2.boundingRect(cont)
                    confidence = box.conf.item()  # Confidence score for the prediction

                    # Draw bounding box and label with confidence
                    # cv2.rectangle(img, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
                    cv2.polylines(img, [cont], True, (255, 0, 0), 2)
                    cv2.putText(img, f"{c} {confidence:.2f}", (x, y - 10), cv2.FONT_ITALIC, 0.5, (0, 0, 255), 2)

    # Display the processed frame
    cv2.imshow("Detection", img)

    # Maintain the desired frame rate
    elapsed_time = time.time() - start_time
    delay = max(1, int((1 / desired_frame_rate - elapsed_time) * 1000))
    if cv2.waitKey(delay) == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
