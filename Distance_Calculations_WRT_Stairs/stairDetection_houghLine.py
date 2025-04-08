import cv2
import numpy as np

# Load and resize the image
image = cv2.imread("stair.jpg")
resized_image = cv2.resize(image, (800, 600))  # Resize for easier processing

# Convert to HSV color space for color detection
hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

# Define HSV range for red color (adjust if needed)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# Create masks for red color
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = mask1 | mask2

# Find contours in the red mask
contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Process contours to find the marker and approximate stair edge
stair_edge_x = None
stair_edge_y = None

if contours:
    # Find the largest red contour (assuming it's the marker)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Draw a rectangle around the detected red marker for visualization
    cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Set approximate x and y coordinates of the stair edge based on marker position
    stair_edge_x = x + w // 2
    stair_edge_y = y + h

    # Define a larger region around the marker to capture the entire stair edge
    expanded_x1 = max(0, x - 100)  # Extend left
    expanded_x2 = min(resized_image.shape[1], x + w + 100)  # Extend right
    expanded_y1 = y - 30
    expanded_y2 = y + 80

    # Extract the region of interest
    region_of_interest = resized_image[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
    
    # Convert region to grayscale
    gray_roi = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges_roi = cv2.Canny(gray_roi, 50, 150)

    # Use Hough Transform to detect lines
    lines = cv2.HoughLinesP(edges_roi, 1, np.pi / 180, threshold=20, minLineLength=80, maxLineGap=10)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw detected lines in the region for visualization
            cv2.line(region_of_interest, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Place the region with detected lines back in the main image for display
    resized_image[expanded_y1:expanded_y2, expanded_x1:expanded_x2] = region_of_interest

# Display the results
cv2.imshow("Red Marker Detection and Enhanced Edge Detection", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
