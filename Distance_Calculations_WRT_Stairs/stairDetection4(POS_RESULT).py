import cv2
import numpy as np

# Set pixel-to-centimeter conversion factor (manually define based on known reference)
conversion_factor = 0.03 # Example: 1 pixel = 0.0264 cm (adjust this based on your setup)

# Load the image
image = cv2.imread("stair1.jpg")

# Convert the image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the color range for red (tweak values based on the red mark in the image)
lower_red = np.array([0, 120, 70])  # Lower range of red
upper_red = np.array([10, 255, 255])  # Upper range of red

# Create a mask for the red mark
mask = cv2.inRange(hsv, lower_red, upper_red)

# Find contours for the red mark
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Extract the center of the largest red mark contour
if contours:
    red_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(red_contour)
    if M["m00"] != 0:
        red_mark = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    else:
        red_mark = None
else:
    print("No red mark detected!")
    red_mark = None

# Detect edges of the stair
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# Detect contours of the stair edges
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

if red_mark and contours:
    # Find the horizontal stair edge (step) closest vertically to the red mark
    selected_contour = None
    for contour in contours:
        for point in contour:
            x, y = point[0]
            if abs(y - red_mark[1]) < 20:  # Look for contours near the red mark's vertical position
                selected_contour = contour
                break
        if selected_contour is not None:
            break

    if selected_contour is not None:
        # Identify the leftmost and rightmost points on the detected stair step
        x_coords = selected_contour[:, 0, 0]
        leftmost = (min(x_coords), red_mark[1])
        rightmost = (max(x_coords), red_mark[1])

        # Determine which endpoint is closest to the red mark horizontally
        if abs(leftmost[0] - red_mark[0]) < abs(rightmost[0] - red_mark[0]):
            endpoint = leftmost
        else:
            endpoint = rightmost

        # Calculate the horizontal distance in pixels
        distance_pixels = abs(red_mark[0] - endpoint[0])

        # Convert the distance to centimeters
        distance_cm = distance_pixels * conversion_factor

        # Annotate the image
        output_image = image.copy()
        cv2.circle(output_image, red_mark, 10, (0, 0, 255), -1)  # Mark the red point
        cv2.circle(output_image, endpoint, 10, (255, 0, 0), -1)  # Mark the endpoint
        cv2.line(output_image, red_mark, endpoint, (0, 255, 0), 3)  # Draw the connecting line

        # Add text to indicate the distance
        distance_text = f"Distance: {distance_cm:.2f} cm"
        cv2.putText(output_image, distance_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the result
        cv2.imshow("Output", output_image)
        print(f"Distance between red mark and stair endpoint: {distance_cm:.2f} cm")

        # Wait for a key press to close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Could not find the stair step near the red mark.")
else:
    print("Unable to detect red mark or stair edges.")
