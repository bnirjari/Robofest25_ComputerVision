import cv2
import numpy as np

def nothing(x):
    pass

# Create a window for trackbars
cv2.namedWindow('Color Calibration')

# Create trackbars for color detection
cv2.createTrackbar('H Low', 'Color Calibration', 0, 179, nothing)
cv2.createTrackbar('S Low', 'Color Calibration', 0, 255, nothing)
cv2.createTrackbar('V Low', 'Color Calibration', 0, 255, nothing)
cv2.createTrackbar('H High', 'Color Calibration', 179, 179, nothing)
cv2.createTrackbar('S High', 'Color Calibration', 255, 255, nothing)
cv2.createTrackbar('V High', 'Color Calibration', 255, 255, nothing)

# Open camera
cap = cv2.VideoCapture(0)

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Get trackbar positions
    h_low = cv2.getTrackbarPos('H Low', 'Color Calibration')
    s_low = cv2.getTrackbarPos('S Low', 'Color Calibration')
    v_low = cv2.getTrackbarPos('V Low', 'Color Calibration')
    h_high = cv2.getTrackbarPos('H High', 'Color Calibration')
    s_high = cv2.getTrackbarPos('S High', 'Color Calibration')
    v_high = cv2.getTrackbarPos('V High', 'Color Calibration')
    
    # Define range of color in HSV
    lower_red = np.array([h_low, s_low, v_low])
    upper_red = np.array([h_high, s_high, v_high])
    
    # Create mask
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # Bitwise-AND mask and original image
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Show the images
    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)
    
    # Break loop
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC key
        break

# Release resources
cap.release()
cv2.destroyAllWindows()