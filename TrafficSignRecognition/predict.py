
import os
import cv2
import numpy as np
import joblib

kernele3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernele5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "traffic_sign_model.pkl"
)

# Thay bằng đường dẫn ảnh bạn muốn kiểm tra
IMAGE_PATH = os.path.join(
    BASE_DIR,
    "52400174_Output-video1-Frame-413.jpg"
)


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
class_names = model_data["class_names"]
IMAGE_SIZE = model_data["image_size"]
hog_params = model_data["hog_params"]

# print("Model loaded successfully!")
# print("Number of classes:", len(class_names))

# print("=== MODEL INFORMATION ===")
# print("Classes:", class_names)
# print("Number of classes:", len(class_names))
# print("Model classes:", model.classes_)
# print("=========================")

# ==========================================
# 3. INITIALIZE HOG
# ==========================================

hog = cv2.HOGDescriptor(
    hog_params["winSize"],
    hog_params["blockSize"],
    hog_params["blockStride"],
    hog_params["cellSize"],
    hog_params["nbins"]
)


# ==========================================
# 4. IMAGE PREPROCESSING
# ==========================================
def readFromFile(filename):
    capSRC = f"videos/{filename}.mp4"
    return capSRC

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
    
    mask = trapezoidMask & rectangleMask
    
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

def fillAndHealMask(binary_mask, min_area = 400, max_area = 60000, min_solidity = 0.70):
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
        epsilon = 0.025 * perimeter
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
        if not 0.1 <= ratio <= 4.5:
            continue

        boxes.append((x, y, w, h))  # Lưu lại tọa độ box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    return boxes  # Trả về danh sách tọa độ
        
def filterTri(contours, image, min_angle=25, max_angle=130, limit=350):
    boxes = []  # Thêm danh sách lưu tọa độ
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < limit:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        # Xấp xỉ contour
        epsilon = 0.06 * perimeter
        approx = cv2.approxPolyDP(
            contour, epsilon, True
        )

        # Tam giác có 3 đỉnh
        if len(approx) != 3:
            continue

        if not isTriangle(approx, min_angle, max_angle):
            continue

        x, y, w, h = cv2.boundingRect(approx)
        if w < 15 or h < 15:
            continue

        ratio = w / h
        if not 0.5 <= ratio <= 1.5:
            continue

        boxes.append((x, y, w, h))  # Lưu lại tọa độ box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    return boxes  # Trả về danh sách tọa độ

def extractSignROI(frame, box, padding=0.15):

    x, y, w, h = box

    height, width = frame.shape[:2]

    # Calculate padding
    pad_x = int(w * padding)
    pad_y = int(h * padding)

    # Expand bounding box
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)

    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)

    # Crop original image
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    return roi

def preprocessImage(image, size=IMAGE_SIZE):

    h, w = image.shape[:2]

    scale = size / max(h, w)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    interpolation = (
        cv2.INTER_AREA if scale < 1
        else cv2.INTER_LINEAR
    )

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=interpolation
    )

    canvas = np.zeros(
        (size, size, 3),
        dtype=np.uint8
    )

    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    gray = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2GRAY
    )

    return gray


# ==========================================
# 5. PREDICT TRAFFIC SIGN
# ==========================================

def predictSign(image):
    if image is None or image.size == 0:
        return None, 0.0, 0.0

    processed = preprocessImage(image)
    features = hog.compute(processed)
    if features is None:
        return None, 0.0, 0.0

    features = features.reshape(1, -1)
    prediction = model.predict(features)[0]
    scores = model.decision_function(features)

    if len(class_names) == 2:
        top1 = abs(float(scores[0]))
        gap = top1
    else:
        scores_1d = scores[0]
        sorted_scores = np.sort(scores_1d)[::-1]
        top1 = float(sorted_scores[0])
        top2 = float(sorted_scores[1])
        gap = top1 - top2  # Độ tự tin vượt trội so với lớp đứng thứ 2

    return prediction, top1, gap

def predict_on_frame(
    clean_frame,
    vis_frame,
    blu_contours,
    redblu_contours,
    redyel_contours,
    conf_threshold=15,
    nms_threshold=0.35,
    gap_threshold=0.8,
):
    # Dùng frame nháp để tránh các hàm filter* vẽ khung xanh sớm trước khi lọc ngưỡng
    temp_frame = vis_frame.copy()

    # Hạ mức limit diện tích cho khớp với ngưỡng w, h bên trong các hàm filter
    rec_boxes = filterRec(blu_contours, temp_frame, min_angle=80, max_angle=110, limit=650)
    cir_boxes = filterCir(redblu_contours, temp_frame, min_circularity=0.85, limit=650)
    tri_boxes = filterTri(redyel_contours, temp_frame, min_angle=25, max_angle=130, limit=350)

    all_boxes = rec_boxes + cir_boxes + tri_boxes
    if not all_boxes:
        return []

    candidate_boxes = []
    candidate_labels = []
    candidate_scores = []

    # 1. Phân loại từng vùng ứng viên và lọc theo margin/score
    for box in all_boxes:
        roi = extractSignROI(clean_frame, box, padding=0.15)
        if roi is None or roi.size == 0:
            continue

        label, top1_score, margin_gap = predictSign(roi)

        if label is None or top1_score < conf_threshold or margin_gap < gap_threshold:
            continue

        x, y, w, h = map(int, box)
        score_val = float(top1_score)
        candidate_boxes.append([x, y, w, h])
        candidate_labels.append(label)
        candidate_scores.append(score_val)

    if not candidate_boxes:
        return []

    # 2. Áp dụng Non-Maximum Suppression (NMS) để loại bỏ các khung trùng lặp/lồng nhau
    indices = cv2.dnn.NMSBoxes(
        bboxes=candidate_boxes,
        scores=candidate_scores,
        score_threshold=conf_threshold,
        nms_threshold=nms_threshold,
    )

    detections = []
    if len(indices) > 0:
        for idx in indices.flatten():
            idx = int(idx)
            x, y, w, h = candidate_boxes[idx]
            label = candidate_labels[idx]
            score_val = candidate_scores[idx]

            detections.append(((x, y, w, h), label, score_val))

            # Chỉ vẽ lên vis_frame sau khi đã qua cả 2 bước lọc
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{label} ({score_val:.2f})"
            cv2.putText(
                vis_frame,
                text,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    return detections


# các đoạn màu

# đỏ
# đỏ 1
lower_red1 = np.array([0, 60, 40]) # personally, np.array([0, 60, 40])
upper_red1 = np.array([5, 255, 255]) #             np.array([5, 255, 255])
# đỏ 2
lower_red2 = np.array([160, 60, 40]) # personally, np.array([160, 60, 40])
upper_red2 = np.array([180, 255, 255]) #             np.array([180, 255, 255])

# vàng
lower_yellow = np.array([12, 110, 90])
upper_yellow = np.array([28, 255, 255])

# xanh dương
lower_blue = np.array([105, 140, 70]) # personally, np.array([105, 120, 45]),
upper_blue = np.array([125, 255, 255]) #            np.array([115, 255, 255])


# ==========================================
# 6. TEST WITH A SINGLE IMAGE
# ==========================================

def main():
    image = cv2.imread(IMAGE_PATH)

    # Kiểm tra ảnh hợp lệ ngay sau khi đọc
    if image is None:
        print("[ERROR] Cannot load test image!")
        print("Image path:", IMAGE_PATH)
        return

    clean_image = image.copy()
    height, width = image.shape[:2]
    
    cv2.namedWindow("Traffic Sign Prediction", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Traffic Sign Prediction", int(width * 0.3), int(height * 0.3))

    blindfold = createEvilROI(height, width)

    blur = cv2.GaussianBlur(image, (5, 5), 0)
    blur_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    red1 = getColorMask(blur_hsv, lower_red1, upper_red1)
    red2 = getColorMask(blur_hsv, lower_red2, upper_red2)
    red = (red1 | red2) & blindfold
    yel = getColorMask(blur_hsv, lower_yellow, upper_yellow) & blindfold
    blu = getColorMask(blur_hsv, lower_blue, upper_blue) & blindfold

    red = removeStrays(red, kernele3, 2)
    yel = removeStrays(yel, kernele3, 2)
    blu = removeStrays(blu, kernele3, 2)

    red = mendCracks(red, kernele5, 2)
    yel = mendCracks(yel, kernele5, 2)
    blu = mendCracks(blu, kernele5, 2)

    red = fillAndHealMask(red)
    yel = fillAndHealMask(yel)
    blu = fillAndHealMask(blu)

    rb_collab = red | blu
    ry_collab = red | yel

    blu_contours, _ = cv2.findContours(blu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    redblu_contours, _ = cv2.findContours(rb_collab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    redyel_contours, _ = cv2.findContours(ry_collab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = predict_on_frame(
        clean_image,
        image,
        blu_contours,
        redblu_contours,
        redyel_contours,
        conf_threshold=0.5,
        nms_threshold=0.35,
    )

    # In kết quả để theo dõi điểm confidence thực tế của model
    print(f"Detected {len(detections)} traffic sign(s):")
    for box, label, conf in detections:
        print(f" - Label: {label} | Confidence: {conf:.3f} | Box: {box}")

    cv2.imshow("Traffic Sign Prediction", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()