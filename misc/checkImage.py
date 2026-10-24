import cv2
import numpy as np

color_dict_HSV = {'black1': np.array([180, 255, 30]), 
                'black0': np.array([0, 0, 0]),
                'white1': np.array([180, 18, 255]), 
                'white0': np.array([0, 0, 231]),
                'red11': np.array([180, 255, 255]), 
                'red10': np.array([159, 50, 70]),
                'red00': np.array([9, 255, 255]), 
                'red01': np.array([0, 50, 70]),
                'green1': np.array([89, 255, 255]), 
                'green0': np.array([36, 50, 70]),
                'blue1': np.array([128, 255, 255]), 
                'blue0': np.array([90, 50, 70]),
                'yellow1': np.array([35, 255, 255]), 
                'yellow0': np.array([25, 50, 70]),
                'purple1': np.array([158, 255, 255]), 
                'purple0': np.array([129, 50, 70]),
                'orange1': np.array([24, 255, 255]), 
                'orange0': np.array([10, 50, 70]),
                'gray1': np.array([180, 18, 230]), 
                'gray0': np.array([0, 0, 40])
}


image = cv2.imread("images/frame_raw.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


black_mask = cv2.inRange(
    hsv,
    color_dict_HSV['black0'],
    color_dict_HSV['black1']
)

white_mask = cv2.inRange(
    hsv,
    color_dict_HSV['white0'],
    color_dict_HSV['white1']
)

red1 = cv2.inRange(
    hsv,
    color_dict_HSV['red00'],
    color_dict_HSV['red01']
)

red2 = cv2.inRange(
    hsv,
    color_dict_HSV['red10'],
    color_dict_HSV['red11']
)

red_mask = cv2.bitwise_or(red1, red2)

green_mask = cv2.inRange(
    hsv,
    color_dict_HSV['green0'],
    color_dict_HSV['green1']
)

blue_mask = cv2.inRange(
    hsv,
    color_dict_HSV['blue0'],
    color_dict_HSV['blue1']
)

yellow_mask = cv2.inRange(
    hsv,
    color_dict_HSV['yellow0'],
    color_dict_HSV['yellow1']
)

purple_mask = cv2.inRange(
    hsv,
    color_dict_HSV['purple0'],
    color_dict_HSV['purple1']
)

orange_mask = cv2.inRange(
    hsv,
    color_dict_HSV['orange0'],
    color_dict_HSV['orange1']
)

gray_mask = cv2.inRange(
    hsv,
    color_dict_HSV['gray0'],
    color_dict_HSV['gray1']
)





cv2.imshow("image",image)
cv2.imshow("hsv_image",hsv)

cv2.imshow("black_mask",black_mask)
cv2.imshow("white_mask",white_mask)
cv2.imshow("red_mask",red_mask)
cv2.imshow("green_mask",green_mask)
cv2.imshow("blue_mask",blue_mask)
cv2.imshow("yellow_mask",yellow_mask)
cv2.imshow("orange_mask",orange_mask)
cv2.imshow("purple_mask",purple_mask)
cv2.imshow("gray_mask",gray_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
