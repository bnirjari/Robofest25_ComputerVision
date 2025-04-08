import cv2
import numpy as np

def pixel_to_cm(pixel_distance, cm_per_pixel):
    return pixel_distance * cm_per_pixel

def preprocess_frame(frame):
    # Convert to grayscale and HSV
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return gray, hsv

def detect_red_mark(hsv):
    lower_red = np.array([158, 135, 102])
    upper_red = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if 500 < area < 5000:
            x, y, w, h = cv2.boundingRect(contour)
            return (x + w // 2, y + h // 2)
    return None

def detect_stair_edge(gray):
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        max_x = 0
        rightmost_line = None
        for line in lines:
            for x1, y1, x2, y2 in line:
                if y1 == y2 and x2 > max_x:  # Horizontal lines
                    max_x = x2
                    rightmost_line = ((x1, y1), (x2, y2))
        return rightmost_line[1] if rightmost_line else None
    return None

def main():
    cap = cv2.VideoCapture(1)
    cm_per_pixel = 0.175# Adjust this based on calibration
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray, hsv = preprocess_frame(frame)
        red_mark = detect_red_mark(hsv)
        stair_edge = detect_stair_edge(gray)
        
        if red_mark and stair_edge:
            pixel_distance = abs(red_mark[0] - stair_edge[0])
            distance_cm = pixel_to_cm(pixel_distance, cm_per_pixel)
            
            # Draw markers and distance
            cv2.circle(frame, red_mark, 5, (0, 0, 255), -1)
            cv2.circle(frame, stair_edge, 5, (0, 255, 0), -1)
            cv2.line(frame, red_mark, stair_edge, (255, 0, 0), 2)
            cv2.putText(frame, f"Distance: {distance_cm:.2f} cm", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Distance Measurement", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
