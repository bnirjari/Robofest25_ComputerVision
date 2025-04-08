import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = 'stair.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Increase contrast and brightness
alpha = 1.5  # Contrast control
beta = 30    # Brightness control
adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

# Apply Canny Edge Detection
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(adjusted, low_threshold, high_threshold)

# Define a region of interest (ROI) to focus on the marked edge
height, width = edges.shape
mask = np.zeros_like(edges)
polygon = np.array([[
    (width // 4, height // 3),  # adjust as needed
    (width * 3 // 4, height // 3),
    (width * 3 // 4, height * 3 // 4),
    (width // 4, height * 3 // 4),
]], np.int32)
cv2.fillPoly(mask, polygon, 255)
roi = cv2.bitwise_and(edges, mask)

# Use Hough Line Transform to detect lines in the ROI
lines = cv2.HoughLinesP(roi, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)

# Draw detected lines on the original image
line_image = np.copy(image)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)//change the rgb to hsv here 

# Display results
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(1, 3, 2), plt.imshow(edges, cmap='gray'), plt.title('Canny Edge Detection')
plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)), plt.title('Detected Edge with Hough Lines')
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()
