import cv2
import numpy as np

# Initialize the webcam feed
cap = cv2.VideoCapture(1)

# Variables to hold points for distance calculation
point1 = None
point2 = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Draw the points if they are selected
    if point1 is not None:
        cv2.circle(frame, point1, 5, (0, 255, 0), -1)
    if point2 is not None:
        cv2.circle(frame, point2, 5, (0, 255, 0), -1)
        
        # Draw the line connecting the points
        cv2.line(frame, point1, point2, (0, 255, 0), 2)
        
        # Calculate the horizontal distance (in pixels)
        horizontal_distance = abs(point2[0] - point1[0])
        cv2.putText(frame, f"Distance: {horizontal_distance}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Real-time Horizontal Distance", frame)

    # Wait for user input to select points
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') and point1 is None:
        point1 = (100, 100)  # Example point 1
    elif key == ord('t') and point2 is None:
        point2 = (400, 100)  # Example point 2

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
