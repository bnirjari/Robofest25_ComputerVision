import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load the model and set up the video
model = YOLO("best5.pt")
class_names = model.names
cap = cv2.VideoCapture("pothole.mp4")

desired_frame_rate = 30
frame_skip = int(cap.get(cv2.CAP_PROP_FPS) / desired_frame_rate)

# Calibration factor: Pixels per cm
pixels_per_cm = 10  # Adjust this value based on your calibration

while True:
    start_time = time.time()
    ret, img = cap.read()
    if not ret:
        break

    if frame_skip > 1:
        for _ in range(frame_skip - 1):
            cap.read()

    img = cv2.resize(img, (640, 480))
    h, w = img.shape[:2]

    # Run the model
    res = model.predict(img, conf=0.5)

    for r in res:
        boxes = r.boxes
        masks = r.masks

        if masks is not None:
            masks = masks.data.cpu()
            for seg, box in zip(masks.cpu().numpy(), boxes):
                seg = cv2.resize(seg, (w, h))
                contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                i = 0
                for cont in contours:
                    M = cv2.moments(cont)
                    if i == 0:  # Ensure only one center per pothole
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.circle(img, (cx, cy), 3, (255, 0, 0), -1)

                            # Distacne from centroid to white line
                            pixel_distance = abs(cx - 320)
                            print("Center: (", cx, ",", cy, ")", "Distance: ", pixel_distance, "px")
                            if pixel_distance <50:
                                if cx < 320:
                                    print("CAUTION!!: Pothole on the left")
                                else:
                                    print("CAUTION!!: Pothole on the right")
                        i = 1

                    # Pothole edge
                    cv2.polylines(img, [cont], True, (0, 255, 0), 2)
                    d = int(box.cls)
                    c = class_names[d]
                    confidence = box.conf.item()
                    x, y, _, _ = cv2.boundingRect(cont)
                    cv2.putText(img, f"{c}", (x, y - 10), cv2.FONT_ITALIC, 0.5, (0, 0, 255), 2)

                    # Draw the white vertical line
                    cv2.circle(img, (320, 240), 3, (255, 255, 255), -1)
                    cv2.line(img, (320, 0), (320, 480), (255, 255, 255), 2)

    # Show the results
    cv2.imshow("Pothole Detection with Centers", img)

    elapsed_time = time.time() - start_time
    delay = max(1, int((1 / desired_frame_rate - elapsed_time) * 1000))
    if cv2.waitKey(delay) == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
