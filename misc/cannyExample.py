# =============================
# lib
# =============================
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================
# def
# =============================
def canny(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    img_canny = cv2.Canny(img_blur, 50, 150)

    return img_canny

def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([
    [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def display_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_img

def make_coordinates(img, line_parameters):
    slope, intercept = line_parameters
    y1 = img.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(img, lines):
    if lines is None:
        return None

    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x1 == x2:
            continue
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]

        if abs(slope) < 0.3:
            continue

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    averaged_lines = []
    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        averaged_lines.append(make_coordinates(img, left_fit_average))

    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)
        averaged_lines.append(make_coordinates(img, right_fit_average))

    return np.array(averaged_lines) if len(averaged_lines) > 0 else None


# =============================
# code
# =============================
# img = cv2.imread('Lab10_lane.jpg',cv2.IMREAD_COLOR)
# lane_img = np.copy(img)

# canny_img = canny(lane_img) # automatically get canny img
# cropped_img = region_of_interest(canny_img) # get only the important part
# lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# averaged_lines = average_slope_intercept(lane_img, lines)
# line_img = display_lines(lane_img, averaged_lines)

# combo_img = cv2.addWeighted(lane_img, 0.8, line_img, 1, 1)

# cv2.imshow("combo_img", combo_img)

# cv2.waitKey()
# cv2.destroyAllWindows()

cap = cv2.VideoCapture('videos/video1.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cv2.namedWindow("Current Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Current Frame", int(width * 0.22), int(height * 0.22))


while(cap.isOpened()):
    _, frame = cap.read()
    if not _:
        break
    canny_img = canny(frame)
    # cropped_img = region_of_interest(canny_img)
    # lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    # averaged_lines = average_slope_intercept(frame, lines)
    # line_img = display_lines(frame, averaged_lines)
    # combo_img = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
    cv2.imshow('Current Frame',canny_img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()