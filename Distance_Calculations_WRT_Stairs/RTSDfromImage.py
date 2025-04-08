import cv2
import numpy as np

def calculate_distance(frame, calibration_factor=0.1):
    # Convert to the HSV color space (better for color detection)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for detecting the red color
    lower_red = np.array([0, 50, 50])  # Adjust if needed
    upper_red = np.array([10, 255, 255])

    # Create a mask to detect the red areas
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours to detect the red mark
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Variables to store the red mark center and the stair edge
    red_mark_center = None
    stair_edge_x = None

    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Filter small noise
            continue

        # Get the bounding box of the red mark
        x, y, w, h = cv2.boundingRect(contour)

        # Find the centroid of the red mark
        red_mark_center = (x + w // 2, y + h // 2)  # Center of the bounding box
        break  # Assuming only one red mark to detect

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Detect horizontal lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is not None:
        # Assume the last detected horizontal line corresponds to the stair edge
        for line in lines:
            for x1, y1, x2, y2 in line:
                if y1 == y2:  # Horizontal line
                    stair_edge_x = max(x1, x2)  # Rightmost point of the stair edge
                    break

    # Calculate the distance
    if red_mark_center and stair_edge_x is not None:
        red_mark_x, _ = red_mark_center
        distance_pixels = abs(red_mark_x - stair_edge_x)
        distance_cm = distance_pixels * calibration_factor

        # Annotate the frame
        cv2.circle(frame, red_mark_center, 5, (0, 0, 255), -1)  # Red mark center
        cv2.line(frame, red_mark_center, (stair_edge_x, red_mark_center[1]), (0, 255, 0), 2)  # Green line
        cv2.putText(frame, f"Distance: {distance_cm:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

# Open webcam feed
cap = cv2.VideoCapture(1)  # Change to the correct camera index if needed

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Process the frame to calculate the distance
    processed_frame = calculate_distance(frame)

    # Display the frame
    cv2.imshow("Real-Time Stair Distance Detection", processed_frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
