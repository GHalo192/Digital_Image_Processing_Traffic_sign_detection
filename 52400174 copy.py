# ==============================
# Định nghĩa các thư viện
# ==============================

import os
import cv2
import numpy as np
import joblib
from collections import defaultdict

# ==============================
# định nghĩa các biến toàn cục
# ==============================

# video name
vid1 = "video1"
vid2 = "video2"

# model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
MODEL_PATH = os.path.join(
    BASE_DIR,
    "52400174_TD_Model.pkl"
)

# load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
class_names = model_data["class_names"]
IMAGE_SIZE = model_data["image_size"]
hog_params = model_data["hog_params"]

# innit HOG
hog = cv2.HOGDescriptor(
    hog_params["winSize"],
    hog_params["blockSize"],
    hog_params["blockStride"],
    hog_params["cellSize"],
    hog_params["nbins"]
)


# kernels
KERNELS3 = np.ones((3, 3), np.uint8)
KERNELS5 = np.ones((5, 5), np.uint8)
KERNELS7 = np.ones((7, 7), np.uint8)
KERNELS9 = np.ones((9, 9), np.uint8)

KERNELE3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
KERNELE5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
KERNELE7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
KERNELE9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

KERNELT5 = np.array([
    [0,0,1,0,0],
    [0,0,1,0,0],
    [0,1,1,1,0],
    [0,1,1,1,0],
    [1,1,1,1,1],
], dtype=np.uint8)
KERNELT7 = np.array([
    [0,0,0,1,0,0,0],
    [0,0,1,1,1,0,0],
    [0,0,1,1,1,0,0],
    [0,1,1,1,1,1,0],
    [0,1,1,1,1,1,0],
    [0,1,1,1,1,1,0],
    [1,1,1,1,1,1,1],
], dtype=np.uint8)

CIRCLE_CLASSES = {
    "Cam_Dau_Xe",
    "Cam_Dung_Xe",
    "Cam_Nguoc_Chieu",
    "Cam_Queo_Trai",
    "Huong_Di_Ben_Phai",
    "Queo_Phai_O_Day",
    "Dung_Xe",
}

TRIANGLE_CLASSES = {
    "Di_Cham",
    "Tre_Em_Qua_Duong",
    "Nhuong_Duong",
}

RECT_CLASSES = {
    "Chi_Dan_Duong",
    "Khu_Vuc_Dau_Xe",
    "Khu_Vuc_Nguoi_Qua_Duong",
}

# Các đoạn màu HSV
lower_red1 = np.array([0, 60, 40])
upper_red1 = np.array([5, 255, 255])
lower_red2 = np.array([160, 60, 40])
upper_red2 = np.array([180, 255, 255])

lower_yellow = np.array([12, 110, 90])
upper_yellow = np.array([28, 255, 255])

lower_blue = np.array([105, 140, 70])
upper_blue = np.array([125, 255, 255])

# ==============================
# Định nghĩa các hàm
# ==============================

# Đọc từ file
def readFromFile(filename):
    return os.path.join(VIDEO_DIR, f"{filename}.mp4")

def medianFilter_Iteration(*masks, iterations=1, s=3):
    processed_masks = []
    
    for m in masks:
        for _ in range(iterations):
            m = cv2.medianBlur(m, ksize=s)
        processed_masks.append(m)
        
    return processed_masks


def computeIOU(boxA, boxB):
    # box dạng (x, y, w, h)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    return inter_area / float(areaA + areaB - inter_area)

# Dùng để cho chương trình "nhớ" được các biển báo trên màn hình
class SignTracker:
    def __init__(self, fps=30, hold_seconds=0.75, iou_threshold=0.25, min_hits=2):
        self.max_age = max(1, int(fps * hold_seconds))
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        # detections: list các tuple ((x, y, w, h), label, confidence) từ predictOnFrame

        matched_track_indices = set()
        matched_det_indices = set()

        # Khớp các detection mới với các track đang tồn tại dựa trên IoU cao nhất
        for d_idx, (det_box, det_label, det_conf) in enumerate(detections):
            best_iou = self.iou_threshold
            best_t_idx = -1

            for t_idx, track in enumerate(self.tracks):
                if t_idx in matched_track_indices:
                    continue
                iou = computeIOU(det_box, track["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_t_idx = t_idx

            if best_t_idx != -1:
                matched_track_indices.add(best_t_idx)
                matched_det_indices.add(d_idx)
                track = self.tracks[best_t_idx]

                # Làm mượt tọa độ box (EMA) để khung không bị giật
                alpha = 0.65
                ox, oy, ow, oh = track["box"]
                nx, ny, nw, nh = det_box
                track["box"] = (
                    int(alpha * nx + (1 - alpha) * ox),
                    int(alpha * ny + (1 - alpha) * oy),
                    int(alpha * nw + (1 - alpha) * ow),
                    int(alpha * nh + (1 - alpha) * oh),
                )

                # Reset bộ đếm thời gian đọng & tăng số lần bắt trúng
                track["missed_frames"] = 0
                track["hits"] += 1

                # Cộng dồn điểm cho nhãn để ghi nhớ ổn định
                track["label_scores"][det_label] += det_conf

 

        # Tạo track mới cho các biển báo vừa xuất hiện
        for d_idx, (det_box, det_label, det_conf) in enumerate(detections):
            if d_idx not in matched_det_indices:
                scores = defaultdict(float)
                scores[det_label] = det_conf

                self.tracks.append({
                    "id": self.next_id,
                    "box": det_box,
                    "label_scores": scores,
                    "best_conf": det_conf,
                    "hits": 1,
                    "missed_frames": 0,
                })
                self.next_id += 1

        # Tăng tuổi cho các track bị mất dấu tạm thời và loại bỏ track quá hạn (0.5s - 1s)
        active_tracks = []
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_track_indices:
                track["missed_frames"] += 1

            if track["missed_frames"] <= self.max_age:
                active_tracks.append(track)

        self.tracks = active_tracks

        # Trả về danh sách các biển báo đủ độ tin cậy để hiển thị
        results = []
        for track in self.tracks:
            if track["hits"] >= self.min_hits:
                # Lấy nhãn có tổng điểm cao nhất trong lịch sử của box này
                remembered_label = max(track["label_scores"].items(), key=lambda x: x[1])[0]
                results.append({
                    "id": track["id"],
                    "box": track["box"],
                    "label": remembered_label,
                    "conf": track["best_conf"],
                    "is_holding": track["missed_frames"] > 0,  # True nếu đang ở trạng thái "đọng lại"
                })
        return results

# Tạo blindfold che màn hình
def createEvilROI(height, width):    
    trapezoidMask = np.ones((height, width), dtype=np.uint8) * 255
    
    rectangleMask = np.ones((height, width), dtype=np.uint8) * 255
    rectangleMask[int(height * 0.6):, :] = 0 
    
    pts = np.array([
        [int(width * 0.01), height],          # Điểm 1: Góc dưới bên trái 
        [int(width * 0.30), int(height * 0.52)], # Điểm 2: Góc trên bên trái 
        [int(width * 0.70), int(height * 0.52)], # Điểm 3: Góc trên bên phải 
        [int(width * 0.99), height]           # Điểm 4: Góc dưới bên phải
    ], dtype=np.int32)
    
    cv2.fillPoly(trapezoidMask, [pts], 0)
    
    mask = trapezoidMask & rectangleMask
    
    return mask

# Tô đặc ruột các hình trên mask
def fillAndHealMask(binary_mask, min_area = 400, max_area = 60000, min_solidity = 0.70):
    # Chỉ lấy các đường viền ngoài cùng (bỏ qua lỗ rỗng bên trong)
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
            
        # Tạo bao lồi (kéo căng dây thun quanh contour để vá vết lõm chữ C)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:
            continue
            
        # Kiểm tra độ đặc (Solidity) để tránh vá nhầm nhiễu nền trời/cây cối
        solidity = area / hull_area
        if solidity >= min_solidity:
            # Tô đặc ruột (FILLED) theo đường bao lồi đã vá
            cv2.drawContours(healed_mask, [hull], -1, 255, thickness=cv2.FILLED)
            
    return healed_mask     

# Xác định là hình vuông / chữ nhật
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

# Xác định là hình tam giác
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

# Lọc tròn
def filterCir(contours, image, min_circularity=0.85, lowerLimit=1200, upperLimit=10000):
    boxes = []  # Thêm danh sách lưu tọa độ
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < lowerLimit:
            continue
        if area > upperLimit:
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

# lọc vuông / chữ nhật
def filterRec(contours, image, min_angle=75, max_angle=105, lowerLimit=2250, upperLimit=40000):
    boxes = []  # Thêm danh sách lưu tọa độ
    for contour in contours:

        area = cv2.contourArea(contour)

        if area < lowerLimit:
            continue
        if area > upperLimit:
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
        if not 0.05 <= ratio <= 4.5:
            continue

        boxes.append((x, y, w, h))  # Lưu lại tọa độ box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    return boxes  # Trả về danh sách tọa độ

# Lọc tam giác
def filterTri(contours, image, min_angle=25, max_angle=130, lowerLimit=600, upperLimit=10000):
    boxes = []  # Thêm danh sách lưu tọa độ
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < lowerLimit:
            continue
        if area > upperLimit:
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
        if w < 15 or h < 15:
            continue
        
        ratio = w / h
        if not 0.5 <= ratio <= 1.5:
            continue

        boxes.append((x, y, w, h))  # Lưu lại tọa độ box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    return boxes  # Trả về danh sách tọa độ

# Crop các bounding box từ hình
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

# xử lý ảnh cho HOG
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

# Phát hiện loại biển báo
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

# Phân loại các loại biển với các loại bounding box
def isShapeValid(shape_type, predicted_label):
    if shape_type == "circle":
        return predicted_label in CIRCLE_CLASSES
    if shape_type == "triangle":
        return predicted_label in TRIANGLE_CLASSES
    if shape_type == "rectangle":
        return predicted_label in RECT_CLASSES
    return False

# Phát hiện các biển báo trên frame 
def predictOnFrame(
    clean_frame,
    vis_frame,
    blu_contours,
    redblu_contours,
    redyel_contours,
    conf_threshold=10,
    nms_threshold=0.35,
    gap_threshold=0.5,
):
    # dùng frame tạm để filter không ghi đè lên frame
    temp_frame = vis_frame.copy()

    rec_boxes = filterRec(blu_contours, temp_frame)
    cir_boxes = filterCir(redblu_contours, temp_frame)
    tri_boxes = filterTri(redyel_contours, temp_frame)

    # Giữ lại thông tin hình dạng contour đi kèm với từng bounding box
    tagged_boxes = (
        [("rectangle", b) for b in rec_boxes]
        + [("circle", b) for b in cir_boxes]
        + [("triangle", b) for b in tri_boxes]
    )

    if not tagged_boxes:
        return []

    candidate_boxes = []
    candidate_labels = []
    candidate_scores = []

    for shape_type, box in tagged_boxes:
        roi = extractSignROI(clean_frame, box, padding=0.15)
        if roi is None or roi.size == 0:
            continue

        label, top1_score, margin_gap = predictSign(roi)

        # Lọc theo độ tự tin và độ chênh lệch điểm với lớp đứng thứ 2
        if label is None or top1_score < conf_threshold or margin_gap < gap_threshold:
            continue

        # Lọc theo ràng buộc hình dạng: loại ngay nếu contour và nhãn mâu thuẫn
        if not isShapeValid(shape_type, label):
            continue

        x, y, w, h = map(int, box)
        candidate_boxes.append([x, y, w, h])
        candidate_labels.append(label)
        candidate_scores.append(float(top1_score))

    if not candidate_boxes:
        return []

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
            detections.append(((x, y, w, h), candidate_labels[idx], candidate_scores[idx]))

    return detections

# Xem video
def watchVideo(vid):
    capSRC = readFromFile(vid)
    cap = cv2.VideoCapture(capSRC)

    if not cap.isOpened():
        raise ValueError("Cannot open video!")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # Giá trị dự phòng nếu file video không đọc được FPS

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Video received successfully!")
    print(f"fps: {fps} | Height: {height} | Width: {width}")
    print("Notes: SPACE/p = Pause | ESC/q = Exit | c = Save frame")    
    
    # Bottom blindfold
    blindfold = createEvilROI(height, width)

    # KHỞI TẠO TRACKER TRƯỚC VÒNG LẶP
    # hold_seconds=0.5: giữ khung đọng lại 0.5 giây sau khi biển báo mất dấu
    # min_hits=2: phải bắt được ít nhất 2 frame liên tiếp mới hiển thị (chống nhiễu nháy 1 frame)
    tracker = SignTracker(fps=fps, hold_seconds=0.5, iou_threshold=0.15, min_hits=2)    
    
    cv2.namedWindow("Current Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Current Frame", int(width * 0.4), int(height * 0.4))
    cv2.namedWindow("B_Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("B_Mask", int(width * 0.3), int(height * 0.3))
    cv2.namedWindow("RB_Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RB_Mask", int(width * 0.3), int(height * 0.3))
    cv2.namedWindow("RY_Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RY_Mask", int(width * 0.3), int(height * 0.3))    
    
    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            # KIỂM TRA ret TRƯỚC KHI .copy() ĐỂ TRÁNH CRASH CUỐI VIDEO
            if not ret:
                print("End of video.")
                break

            clean_frame = frame.copy()

            blur = cv2.GaussianBlur(frame, (5, 5), 0)
            blur_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

            red1 = cv2.inRange(blur_hsv,lower_red1,upper_red1)
            red2 = cv2.inRange(blur_hsv,lower_red2,upper_red2)
            red = (red1 | red2) & blindfold
            yel = cv2.inRange(blur_hsv,lower_yellow,upper_yellow) & blindfold
            blu = cv2.inRange(blur_hsv,lower_blue,upper_blue) & blindfold
            
            #
            # adjusting masks
            # remove stray pixels
            # red = cv2.medianBlur(red,ksize=3) 
            # blu = cv2.medianBlur(blu,ksize=3)
            # yel = cv2.medianBlur(yel,ksize=3)
            red, blu, yel = medianFilter_Iteration(red, blu, yel, iterations=4,s=3)
            # remove stray pixels

            red = cv2.morphologyEx(
                red,
                cv2.MORPH_CLOSE,
                KERNELS5
            )

            blu = cv2.morphologyEx(
                blu,
                cv2.MORPH_CLOSE,
                KERNELS5
            )

            yel = cv2.morphologyEx(
                yel,
                cv2.MORPH_CLOSE,
                KERNELS5
            )
            
        
            # đưa blindfold lên che hết 1/3 dưới của màn hình
            red = red & blindfold
            blu = blu & blindfold
            yel = yel & blindfold
            # đưa blindfold lên che hết 1/3 dưới của màn hình        
        
            red = fillAndHealMask(red)
            yel = fillAndHealMask(yel)
            blu = fillAndHealMask(blu)
                    
            rb_collab = red | blu
            ry_collab = red | yel
            
            # dilate masks
            blu = cv2.dilate(blu,KERNELS5,iterations=1)
            rb_collab = cv2.dilate(rb_collab,KERNELE5,iterations=1)
            ry_collab = cv2.dilate(ry_collab,KERNELT7,iterations=5)
            # dilate masks
                        
            blu_contours, _ = cv2.findContours(blu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            redblu_contours, _ = cv2.findContours(rb_collab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            redyel_contours, _ = cv2.findContours(ry_collab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Lấy danh sách biển báo thô trên frame hiện tại
            detections = predictOnFrame(
                clean_frame,
                frame,
                blu_contours,
                redblu_contours,
                redyel_contours,
                conf_threshold=10,
                nms_threshold=0.35,
                gap_threshold=0.5,
            )

            # CẬP NHẬT TRACKER & VẼ KẾT QUẢ ĐÃ GHI NHỚ LÊN FRAME
            tracked_signs = tracker.update(detections)

            for item in tracked_signs:
                x, y, w, h = item["box"]
                label = item["label"]
                conf = item["conf"]

                # Xanh lá (0, 255, 0) khi đang bắt trực tiếp; Vàng cam (0, 200, 255) khi đang đọng lại
                color = (0, 200, 255) if item["is_holding"] else (0, 255, 0)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    f"#{item['id']} {label}",
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            # watermark
            cv2.putText(frame, "52400174", (width - 250, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Current Frame", frame)
            cv2.imshow("B_Mask", blu)
            cv2.imshow("RB_Mask", rb_collab)
            cv2.imshow("RY_Mask", ry_collab)



        key = cv2.waitKey(1) & 0xFF

        if key == ord(" ") or key == ord("p"):
            paused = not paused

        elif key == ord("c"):
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            images_dir = os.path.join(BASE_DIR, "images")
            os.makedirs(images_dir, exist_ok=True)  # Tự tạo thư mục images nếu chưa có
            outfilename = os.path.join(images_dir, f"52400174_Output-{vid}-Frame-{frame_number}.jpg")
            if cv2.imwrite(outfilename, frame):
                print(f"Saved: {outfilename}")
            else:
                print("Failed to save image!")

        elif key == ord("q") or key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

# Xử lý và xuất video
def processVideo(vid):
    capSRC = readFromFile(vid)
    cap = cv2.VideoCapture(capSRC)

    if not cap.isOpened():
        raise ValueError("Cannot open video!")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # Giá trị dự phòng nếu file video không đọc được FPS

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Video received successfully!")
    print(f"fps: {fps} | Height: {height} | Width: {width}")
    print("Processing video...")
    
  
        
    # Bottom blindfold
    blindfold = createEvilROI(height, width)

    # KHỞI TẠO TRACKER TRƯỚC VÒNG LẶP
    # hold_seconds=0.5: giữ khung đọng lại 0.5 giây sau khi biển báo mất dấu
    # min_hits=2: phải bắt được ít nhất 2 frame liên tiếp mới hiển thị (chống nhiễu nháy 1 frame)
    tracker = SignTracker(fps=fps, hold_seconds=0.5, iou_threshold=0.15, min_hits=2)    
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(VIDEO_DIR, f"52400174_{vid}_output.mp4")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))        
    while True:
        ret, frame = cap.read()
        # KIỂM TRA ret TRƯỚC KHI .copy() ĐỂ TRÁNH CRASH CUỐI VIDEO
        if not ret:
            print("Video Processed.")
            break

        clean_frame = frame.copy()

        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        blur_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        red1 = cv2.inRange(blur_hsv,lower_red1,upper_red1)
        red2 = cv2.inRange(blur_hsv,lower_red2,upper_red2)
        red = (red1 | red2) & blindfold
        yel = cv2.inRange(blur_hsv,lower_yellow,upper_yellow) & blindfold
        blu = cv2.inRange(blur_hsv,lower_blue,upper_blue) & blindfold
        
        red = cv2.morphologyEx(red,cv2.MORPH_OPEN,KERNELE3,iterations=2)
        yel = cv2.morphologyEx(yel,cv2.MORPH_OPEN,KERNELE3,iterations=2)
        blu = cv2.morphologyEx(blu,cv2.MORPH_OPEN,KERNELE3,iterations=2)

        red = cv2.morphologyEx(red,cv2.MORPH_CLOSE,KERNELE3,iterations=3)
        yel = cv2.morphologyEx(yel,cv2.MORPH_CLOSE,KERNELE3,iterations=3)
        blu = cv2.morphologyEx(blu,cv2.MORPH_CLOSE,KERNELE3,iterations=3)
        
        red = fillAndHealMask(red)
        yel = fillAndHealMask(yel)
        blu = fillAndHealMask(blu)

        rb_collab = red | blu
        ry_collab = red | yel
        
        blu_contours, _ = cv2.findContours(blu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        redblu_contours, _ = cv2.findContours(rb_collab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        redyel_contours, _ = cv2.findContours(ry_collab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lấy danh sách biển báo thô trên frame hiện tại
        detections = predictOnFrame(
            clean_frame,
            frame,
            blu_contours,
            redblu_contours,
            redyel_contours,
            conf_threshold=10,
            nms_threshold=0.35,
            gap_threshold=0.5,
        )

        # CẬP NHẬT TRACKER & VẼ KẾT QUẢ ĐÃ GHI NHỚ LÊN FRAME
        tracked_signs = tracker.update(detections)

        for item in tracked_signs:
            x, y, w, h = item["box"]
            label = item["label"]
            conf = item["conf"]

            # Xanh lá (0, 255, 0) khi đang bắt trực tiếp; Vàng cam (0, 200, 255) khi đang đọng lại
            color = (0, 200, 255) if item["is_holding"] else (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"#{item['id']} {label}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # watermark
        cv2.putText(frame, "52400174", (width - 250, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

        if frame is not None:
            out.write(frame)
                
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    
if __name__ == "__main__":
    while True:
        print("===============MENU===============")
        print("1. Process video 1")
        print("2. Process video 2")
        print("3. Watch video 1")
        print("4. Watch video 2")
        print("0. Quit")
        print("===============MENU===============")
        
        choice = input("Please pick an option: ").strip()
        match choice:
            case "0":
                break
            
            case "1":
                processVideo(vid1)
                break
            
            case "2":
                processVideo(vid2)
                break
            
            case "3":
                watchVideo(vid1)
                break
                    
            case "4":
                watchVideo(vid2)
                break
                    
            case _:
                print("==================================")
                print("Please input a valid choice.")
                print("==================================")
