import cv2 as cv
import numpy as np


img = cv.imread("/home/jarvis/sanitron_ws/red_circle.jpg")

cv.imshow("image", img)

hsv_value = cv.cvtColor(img,cv.COLOR_BGR2HSV)

cv.imshow("hsv image", hsv_value)

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

masking1 = cv.inRange(hsv_value, lower_red1, upper_red1)
masking2 = cv.inRange(hsv_value, lower_red2, upper_red2)

masking_for_red = cv.bitwise_or(masking1,masking2)



# counters ones

ker = np.ones((5,5), np.uint8)
red_masking = cv.morphologyEx(masking_for_red, cv.MORPH_OPEN, ker)
red_masking = cv.morphologyEx(masking_for_red, cv.MORPH_CLOSE, ker)


counters,var = cv.findContours(masking_for_red, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

largest_counter = max(counters, key = cv.contourArea)



#  finding the center
center = cv.moments(largest_counter)
if center["m00"] == 0:
    print("there is no center")

x = int( center["m10"] / center["m00"] )    # int coz for exact value
y = int( center["m01"] / center["m00"] )    # int coz for exact value


# now calculating area of circle
area = cv.contourArea(largest_counter)

copy_img = img.copy()
cv.drawContours(copy_img, [largest_counter], -1, (0,255,0), 2)   # taking green for marking
cv.circle(copy_img, (x,y), 5, (255,0,0), -1)

print(f"x:- {x}, y:- {y}")
print(f"area:- {area}, its in pixels")

cv.imshow("Red circle img", copy_img)
cv.imshow("red masked img", red_masking)

cv.waitKey(0)