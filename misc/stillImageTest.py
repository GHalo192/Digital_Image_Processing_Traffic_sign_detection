import numpy as np
import cv2

# kernels
kernels3 = np.ones((3, 3), np.uint8)
kernels5 = np.ones((5, 5), np.uint8)
kernels7 = np.ones((7, 7), np.uint8)
kernels9 = np.ones((9, 9), np.uint8)

kernele3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernele5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernele7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
kernele9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))






# tạo blindfold che màn hình
def createEvilROI(height, width):    
    # Khởi tạo một mảng toàn số 255 (màu trắng) có cùng kích thước với ảnh
    trapezoidMask = np.ones((height, width), dtype=np.uint8) * 255
    
    rectangleMask = np.ones((height, width), dtype=np.uint8) * 255
    rectangleMask[int(height * 0.6):, :] = 0 

    # Định nghĩa 4 đỉnh của hình thang (Thứ tự: Trái dưới, Trái trên, Phải trên, Phải dưới)
    # Bạn có thể tinh chỉnh các tỷ lệ này tùy theo góc gắn camera trên xe của bạn:
    # - Đáy hình thang (rộng) nằm ở cạnh dưới của ảnh để giữ lại lề đường.
    # - Đỉnh hình thang (hẹp hơn) nằm ở phía trên (gần đường chân trời) để tập trung vào giữa đường.
    
    pts = np.array([
        [int(width * 0.01), height],          # Điểm 1: Góc dưới bên trái (gần mép trái)
        [int(width * 0.30), int(height * 0.52)], # Điểm 2: Góc trên bên trái (ngay dưới đường chân trời)
        [int(width * 0.70), int(height * 0.52)], # Điểm 3: Góc trên bên phải (ngay dưới đường chân trời)
        [int(width * 0.99), height]           # Điểm 4: Góc dưới bên phải (gần mép phải)
    ], dtype=np.int32)
    
    # Tô đen (0) cho vùng bên trong hình thang
    cv2.fillPoly(trapezoidMask, [pts], 0)
    
    mask = cv2.bitwise_and(trapezoidMask, rectangleMask)
    
    return mask

def getColorMask(hsv_image, lower, upper):
    mask = cv2.inRange(
        hsv_image,
        lower,
        upper
    )
    return mask

def removeStrays(mask, kernel, i=1):
    new_mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=i
    )
    return new_mask

def mendCracks(mask, kernel, i=1):
    new_mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=i
    )
    return new_mask


def fillAndHealMask(binary_mask, min_area = 700, max_area = 40000, min_solidity = 0.6):
    # 1. Chỉ lấy các đường viền ngoài cùng (bỏ qua lỗ rỗng bên trong)
    contours, _ = cv2.findContours(
        binary_mask, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Tạo một tấm ảnh đen sạch để vẽ kết quả
    healed_mask = np.zeros_like(binary_mask)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Bỏ qua các đốm nhiễu quá nhỏ hoặc chụm quá lớn
        if area < min_area:
            continue
        if area > max_area:
            continue
            
        # 2. Tạo bao lồi (kéo căng dây thun quanh contour để vá vết lõm chữ C)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:
            continue
            
        # 3. Kiểm tra độ đặc (Solidity) để tránh vá nhầm nhiễu nền trời/cây cối
        solidity = area / hull_area
        if solidity >= min_solidity:
            # Tô đặc ruột (FILLED) theo đường bao lồi đã vá
            cv2.drawContours(healed_mask, [hull], -1, 255, thickness=cv2.FILLED)
            
    return healed_mask        

def isRectangle(approx, min_angle=75, max_angle=105):

    points = approx.reshape(4, 2)

    for i in range(4):

        p1 = points[i - 1] - points[i]
        p2 = points[(i + 1) % 4] - points[i]

        norm1 = np.linalg.norm(p1)
        norm2 = np.linalg.norm(p2)

        if norm1 == 0 or norm2 == 0:
            return False

        cosine = np.dot(p1, p2) / (norm1 * norm2)

        angle = np.degrees(np.arccos(
            np.clip(cosine, -1, 1)
        ))

        if not min_angle <= angle <= max_angle:
            return False

    return True

def isTriangle(approx, min_angle=25, max_angle=130):

    points = approx.reshape(3, 2)

    for i in range(3):

        p1 = points[i - 1] - points[i]
        p2 = points[(i + 1) % 3] - points[i]

        norm1 = np.linalg.norm(p1)
        norm2 = np.linalg.norm(p2)

        if norm1 == 0 or norm2 == 0:
            return False

        cosine = np.dot(p1, p2) / (norm1 * norm2)

        angle = np.degrees(np.arccos(
            np.clip(cosine, -1, 1)
        ))

        if not min_angle <= angle <= max_angle:
            return False

    return True

# def filterCir(contours, image, min_circularity = 0.80,limit=1200): # độ tròn nên là 0.80 hoặc hơn
#     for contour in contours:
#         # Mặc kệ những contour nhỏ hơn 1200 pixels
#         area = cv2.contourArea(contour)

#         if area < limit:
#             continue

#         # Tính độ tròn của contour
#         perimeter = cv2.arcLength(contour, True)

#         if perimeter == 0:
#             continue

#         circularity = (4 * np.pi * area) / (perimeter ** 2)

#         if circularity < min_circularity:
#             continue

#         # Vẽ bounding box
#         x, y, w, h = cv2.boundingRect(contour)

#         if w < 25 or h < 25:
#             continue

#         # Filter by aspect ratio
#         ratio = w / h

#         if not 0.5 <= ratio <= 1.5:
#             continue

#         # Draw bounding box
#         cv2.rectangle(
#             image,
#             (x, y),
#             (x + w, y + h),
#             (0, 255, 0),
#             2
#         ) Old code, will draw contours immediately


def filterCir(contours, image, min_circularity=0.80, limit=1200):
    boxes = []  # Thêm danh sách lưu tọa độ
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < limit:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = (4 * np.pi * area) / (perimeter ** 2)
        
        if circularity < min_circularity:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < 25 or h < 25:
            continue

        ratio = w / h
        if not 0.5 <= ratio <= 1.5:
            continue

        boxes.append((x, y, w, h))  # Lưu lại tọa độ box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    return boxes  # Trả về danh sách tọa độ


def filterRec(contours, image, min_angle=75, max_angle=105, limit=2250):
    boxes = []  # Thêm danh sách lưu tọa độ
    for contour in contours:

        area = cv2.contourArea(contour)

        if area < limit:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        # Xấp xỉ contour
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(
            contour, epsilon, True
        )

        # Hình chữ nhật có 4 đỉnh
        if len(approx) != 4:
            continue

        if not isRectangle(approx, min_angle, max_angle):
            continue        

        x, y, w, h = cv2.boundingRect(approx)
        if w < 25 or h < 25:
            continue

        ratio = w / h
        if not 0.5 <= ratio <= 1.5:
            continue

        boxes.append((x, y, w, h))  # Lưu lại tọa độ box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    return boxes  # Trả về danh sách tọa độ
        
        
def filterTri(contours, image, min_angle=25, max_angle=130, limit=1500):
    boxes = []  # Thêm danh sách lưu tọa độ
    for contour in contours:

        area = cv2.contourArea(contour)

        if area < limit:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        # Xấp xỉ contour
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(
            contour, epsilon, True
        )

        # Tam giác có 3 đỉnh
        if len(approx) != 3:
            continue

        if not isTriangle(approx, min_angle, max_angle):
            continue

        x, y, w, h = cv2.boundingRect(approx)
        if w < 25 or h < 25:
            continue

        ratio = w / h
        if not 0.5 <= ratio <= 1.5:
            continue

        boxes.append((x, y, w, h))  # Lưu lại tọa độ box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    return boxes  # Trả về danh sách tọa độ
        
# def drawBoundingBox(boxes, image):
    
#     cv2.rectangle(
#         image,
#         (x, y),
#         (x + w, y + h),
#         (0, 255, 0),
#         2
#     ) # might make it separate, maybr unneeded

def extractSignROI(frame, box, padding=0.15, size=64):

    x, y, w, h = box

    height, width = frame.shape[:2]

    # Add padding around bounding box
    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)

    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)

    # Crop image
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    # Resize to fixed dimensions
    roi = cv2.resize(
        roi,
        (size, size),
        interpolation=cv2.INTER_AREA
    )

    return roi

# preps before reading the video


frame = cv2.imread("images/52400174_Output-video1-Frame-553.jpg")

clean_frame = frame.copy()

height, width = frame.shape[:2]


    
# Tạo các cửa sổ nhỏ để có thể cùng lúc xem video và 4 masks
cv2.namedWindow("Current Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Current Frame", int(width * 0.22), int(height * 0.22))
cv2.namedWindow("red_mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("red_mask", int(width * 0.22), int(height * 0.22))
cv2.namedWindow("yellow_mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("yellow_mask", int(width * 0.22), int(height * 0.22))    
cv2.namedWindow("blue_mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("blue_mask", int(width * 0.22), int(height * 0.22))
cv2.namedWindow("r_b_collab_mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("r_b_collab_mask", int(width * 0.22), int(height * 0.22))
cv2.namedWindow("r_y_collab_mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("r_y_collab_mask", int(width * 0.22), int(height * 0.22))


# các đoạn màu

# đỏ
# đỏ 1
lower_red1 = np.array([0, 60, 40]) # personally, np.array([0, 60, 40])
upper_red1 = np.array([5, 255, 255]) #             np.array([5, 255, 255])
# đỏ 2
lower_red2 = np.array([160, 60, 40]) # personally, np.array([160, 60, 40])
upper_red2 = np.array([180, 255, 255]) #             np.array([180, 255, 255])

# vàng
lower_yellow = np.array([18, 150, 120]) # might switch to np.array([18, 150, 120])
upper_yellow = np.array([33, 255, 255]) #                 np.array([33, 255, 255])

# xanh dương
lower_blue = np.array([105, 140, 70]) # personally, np.array([105, 120, 45]),
upper_blue = np.array([125, 255, 255]) #            np.array([115, 255, 255])

# xanh lá
lower_green = np.array([35, 80, 60])
upper_green = np.array([85, 255, 255])

# trắng
lower_white = np.array([0, 0, 200]) # is way too likely to target the sky, unused
upper_white = np.array([180, 30, 255])

# bottom blindfold
blindfold = createEvilROI(height, width)




            
blur = cv2.GaussianBlur(frame, (5,5), 0)
# holy hell! GaussianBlur, the GOAT!

# lấy ảnh hsv
blur_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# lấy ảnh hsv    

# mask đỏ
red1 = getColorMask(blur_hsv,lower_red1,upper_red1)
red2 = getColorMask(blur_hsv,lower_red2,upper_red2)
red = red1 | red2

# mask vàng
yel = getColorMask(blur_hsv,lower_yellow,upper_yellow)

# mask xanh dương
blu = getColorMask(blur_hsv,lower_blue,upper_blue)

# mask xanh lá

# pure mask list: red, blu, yel        

# apply blindfold
red = cv2.bitwise_and(red,blindfold) # will switch to use the fancy & symbol later
blu = cv2.bitwise_and(blu,blindfold)
yel = cv2.bitwise_and(yel,blindfold)

# remove stray pixels (MORPH_OPEN)
red = removeStrays(red, kernele3, 2)
yel = removeStrays(yel, kernele3, 2)
blu = removeStrays(blu, kernele3, 2)

red = cv2.dilate(red,kernel=kernele7)
yel = cv2.dilate(yel,kernel=kernele7)
blu = cv2.dilate(blu,kernel=kernele7)

red = cv2.erode(red,kernel=kernele7)
yel = cv2.erode(yel,kernel=kernele7)
blu = cv2.erode(blu,kernel=kernele7)

# attempt re connect (MORPH_CLOSE)
red = mendCracks(red, kernele5, 2)
yel = mendCracks(yel, kernele5, 2)
blu = mendCracks(blu, kernele7, 2)

# fill inside

red = fillAndHealMask(red)
yel = fillAndHealMask(yel)
blu = fillAndHealMask(blu)



# pre_merge

rb_collab = cv2.bitwise_or(red, blu) # will switch to use the fancy | symbol later
ry_collab = cv2.bitwise_or(red, yel)

# post_merge

# red = cv2.erode(red,kernel=kernele5)
# rb_collab = cv2.erode(rb_collab,kernel=kernele5)


blu_contours, _ = cv2.findContours(
    blu,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)    

redblu_contours, _ = cv2.findContours(
    rb_collab,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

redyel_contours, _ = cv2.findContours(
    ry_collab,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


# vẽ bounding boxes
rec_boxes = filterRec(blu_contours, frame, min_angle=80, max_angle=110, limit=2000)
cir_boxes = filterCir(redblu_contours, frame, min_circularity=0.85, limit=1750)
tri_box = filterTri(redyel_contours, frame, min_angle=25, max_angle=130, limit=1750)
# vẽ bounding boxes  

all_boxes = rec_boxes + cir_boxes + tri_box
                
cropped_rois = []

for i, box in enumerate(all_boxes):
    # Gọi hàm extractSignROI có sẵn của bạn trên clean_frame
    roi = extractSignROI(clean_frame, box, padding=0.15, size=64)
    
    if roi is not None:
        cropped_rois.append(roi)
        
        # CÁCH A: Xuất ra file ảnh trên ổ cứng
        cv2.imwrite(f"training/sign_{i}.jpg", roi)
        
        # CÁCH B1: Xem từng ảnh trên từng cửa sổ riêng biệt
        cv2.imshow(f"Sign_{i}", roi)
    
    

cv2.imshow('Current Frame', frame)
cv2.imshow('red_mask', red)
cv2.imshow('yellow_mask', yel)                
cv2.imshow('blue_mask', blu)
cv2.imshow('r_b_collab_mask', rb_collab)
cv2.imshow('r_y_collab_mask', ry_collab)

    # note to self:
    # have pure blue detect recangular signs
    # have red blue collab detect circular signs
    # have red yellow collab detect triangular signs
    
# if len(cropped_rois) > 0:
#     preview_strip = np.hstack(cropped_rois)
#     cv2.imshow("All Extracted Signs", preview_strip)
        
cv2.waitKey(0)
cv2.destroyAllWindows()