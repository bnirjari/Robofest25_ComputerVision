import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("best1.pt")
class_names = model.names
cap = cv2.VideoCapture("pothole.mp4")
cnt = 0

while True:
    ret, img = cap.read()
    if not ret:
        break
    cnt += 1
  

    img = cv2.resize(img, (640,480))
    h, w = img.shape[:2]
    res = model.predict(img)

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
                    x, y, w, h = cv2.boundingRect(cont)
                    cv2.polylines(img, [cont], True, (255, 0, 0), 2)
                    cv2.putText(img, c, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Detection", img)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
