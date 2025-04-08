import cv2
import numpy as np

def detect_red_mark(frame):
    """Detect a very specific shade of red"""
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # VERY NARROW range for a specific red shade
    # Adjust these values precisely based on your exact red mark
    lower_red = np.array([158,135,102])  # Very specific lower bound
    upper_red = np.array([180, 255, 255])  # Very specific upper bound
    
    # Create mask for the exact red shade
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # Optional: Morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and process red mark
    for contour in contours:
        area = cv2.contourArea(contour)
        if 500 < area < 5000:  # Very strict area filtering
            x, y, w, h = cv2.boundingRect(contour)
            return (x + w // 2, y + h // 2)
    
    return None

def detect_stair_edge(frame):
    """Detect the rightmost edge of the stair"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                             minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        # Find the rightmost horizontal line
        max_x = 0
        rightmost_line = None
        for line in lines:
            for x1, y1, x2, y2 in line:
                if y1 == y2 and x2 > max_x:
                    max_x = x2
                    rightmost_line = ((x1, y1), (x2, y2))
        
        return rightmost_line[1] if rightmost_line else None
    
    return None

def main():
    # Open the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(1)
    
    # Calibration factor (adjust based on your specific setup)
    calibration_factor = 0.5# cm per pixel
    
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create a copy for drawing
        display_frame = frame.copy()
        
        # Detect red mark
        red_mark = detect_red_mark(frame)
        
        # Detect stair edge
        stair_edge = detect_stair_edge(frame)
        
        # Calculate distance if both are detected
        if red_mark and stair_edge:
            # Calculate horizontal distance
            distance_pixels = abs(red_mark[0] - stair_edge[0])
            distance_cm = distance_pixels * calibration_factor
            
            # Draw markers and distance
            cv2.circle(display_frame, red_mark, 5, (0, 0, 255), -1)
            cv2.circle(display_frame, stair_edge, 5, (0, 255, 0), -1)
            cv2.line(display_frame, red_mark, stair_edge, (255, 0, 0), 2)
            
            # Display distance
            cv2.putText(display_frame, 
                        f"Distance: {distance_cm:.2f} cm", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow("Precise Red Mark Detection", display_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()