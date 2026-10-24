import cv2
import numpy as np




image = cv2.imread("images/colorSpectrum1.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.imshow("image",image)
cv2.imshow("hsv_image",hsv)


# get masks
# red mask
red1 = cv2.inRange(
    hsv,
    np.array([0, 60, 40]),
    np.array([5, 255, 255])
)
red2 = cv2.inRange(
    hsv,
    np.array([160, 60, 40]),
    np.array([180, 255, 255])
)
red_mask = cv2.bitwise_or(red1, red2)
# red mask


# blue mask
blue_mask = cv2.inRange(
    hsv,
    np.array([105, 120, 45]),
    np.array([115, 255, 255])
)
# blue mask




red = cv2.bitwise_and(image,image,mask=red_mask)
blue = cv2.bitwise_and(image,image,mask=blue_mask)


cv2.imshow("red_mask",red_mask)
cv2.imshow("blue_mask",blue_mask)

cv2.imshow("red",red)
cv2.imshow("blue",blue)





cv2.waitKey(0)
cv2.destroyAllWindows()
