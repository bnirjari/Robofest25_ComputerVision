import cv2
import numpy as np

# Step 1: Read the image
image = cv2.imread(r"bar.png")
if image is None:
    print("Error: Unable to read image!")
    exit()

# Step 2: Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply edge detection
edges = cv2.Canny(gray, 50, 150)

# Step 4: Apply morphological transformations to create a mask
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Step 5: Erode and dilate to clean up the mask
morphed = cv2.erode(morphed, None, iterations=2)
morphed = cv2.dilate(morphed, None, iterations=2)

# Step 6: Find contours
contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 7: Filter the largest contour
if len(contours) == 0:
    print("No barcode detected!")
    exit()

barcode_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(barcode_contour)

# Step 8: Draw the bounding box on the original image
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Step 9: Display the result
cv2.imshow("Detected Barcode", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
