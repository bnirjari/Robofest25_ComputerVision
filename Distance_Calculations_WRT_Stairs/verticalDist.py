import cv2
import numpy as np

# Known width of the ribbon (in centimeters or any unit you prefer)
KNOWN_WIDTH = 5.0  # Adjust this based on the actual width of the ribbon

# Camera calibration (you may need to adjust the focal length based on your setup)
FOCAL_LENGTH = 650  # Approximate focal length in pixels

def calculate_distance(perceived_width):
    """Calculate the distance from the camera to the ribbon based on its perceived width in pixels."""
    if perceived_width > 0:
        return (KNOWN_WIDTH * FOCAL_LENGTH) / perceived_width
    else:
        return None

def detect_ribbon(frame):
    """Detect the red ribbon in the frame and return its bounding box and distance."""
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the color range for detecting red (may need adjustment for lighting)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Assume the largest contour is the ribbon
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # Calculate distance based on the perceived width of the ribbon in pixels
        distance = calculate_distance(w)
        
        # Draw bounding box and display distance
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Distance: {distance:.2f} cm", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display the width and height of the detected object
        cv2.putText(frame, f"Width: {w} px", (x, y + h + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Height: {h} px", (x, y + h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame, distance, w, h
    else:
        return frame, None, None, None

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect ribbon and calculate distance
        frame, distance, width, height = detect_ribbon(frame)

        # Display the frame
        cv2.imshow("Ribbon Detection", frame)
        
        # Print the dimensions and distance on the console (optional)
        if distance is not None:
            print(f"Distance: {distance:.2f} cm")
            print(f"Width: {width} px, Height: {height} px")
        
        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
