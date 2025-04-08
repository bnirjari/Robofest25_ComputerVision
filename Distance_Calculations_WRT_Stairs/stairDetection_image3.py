import cv2
import numpy as np

# Load the image (upload the correct image path here)
image = cv2.imread('stair1.jpg')

# Convert to the HSV color space (better for color detection)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range for detecting the red color (adjust if needed)
lower_red = np.array([0, 50, 50])  # Lower range of red color
upper_red = np.array([10, 255, 255])  # Upper range of red color

# Create a mask to detect the red areas
mask = cv2.inRange(hsv, lower_red, upper_red)

# Find contours to detect the red mark
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Variables to store the red mark center and the rightmost edge of the stair
red_mark_center = None
rightmost_stair_edge = None

# Loop through contours to find the red mark and the rightmost edge of the stair
for contour in contours:
    if cv2.contourArea(contour) < 500:  # Filter small noise
        continue
    
    # Get the bounding box of the red mark
    x, y, w, h = cv2.boundingRect(contour)

    # Find the centroid of the red mark
    red_mark_center = (x + w // 2, y + h // 2)  # Center of the bounding box

# Now, calculate the rightmost edge of the stair
# The rightmost edge can be determined from the horizontal position of the last step
height, width = image.shape[:2]

# For simplicity, assume the rightmost edge is at the last visible step
rightmost_stair_edge = (width, height)  # Rightmost point on the horizontal line

# Alternatively, you could use edge detection or another method to identify the stair's edge
# Example: We can use horizontal line detection (to identify where the last step ends)

# Convert to grayscale for edge detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# Find the horizontal line (you could tweak this to detect the stair's edge better)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Let's assume the last detected horizontal line corresponds to the end of the stair
if lines is not None:
    for line in lines:
        for x1, y1, x2, y2 in line:
            if y1 == y2:  # We are looking for a horizontal line
                rightmost_stair_edge = (x2, y2)  # Set the rightmost point

# Calculate the horizontal distance in pixels
if red_mark_center and rightmost_stair_edge:
    red_mark_x, red_mark_y = red_mark_center
    stair_end_x, stair_end_y = rightmost_stair_edge

    # Calculate the distance in pixels (horizontal direction only)
    distance_pixels = abs(red_mark_x - stair_end_x)
    print(f"Distance in pixels: {distance_pixels}")
    
    # Convert pixels to centimeters (use a known calibration factor)
    calibration_factor = 0.1  # This value depends on your setup
    distance_cm = distance_pixels * calibration_factor
    print(f"Distance in cm: {distance_cm}")

    # Annotate the image with the calculated distance
    cv2.circle(image, red_mark_center, 5, (0, 0, 255), -1)  # Red mark center
    cv2.line(image, red_mark_center, rightmost_stair_edge, (0, 255, 0), 2)  # Green line
    cv2.putText(image, f"Distance: {distance_cm:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the annotated image (no need to save, just show)
    cv2.imshow("Distance Calculation", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
