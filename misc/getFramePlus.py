
import cv2
import numpy as np

vid1 = "video1"
vid2 = "video2"

# kernels
kernels3 = np.ones((3, 3), np.uint8)
kernels5 = np.ones((5, 5), np.uint8)
kernels7 = np.ones((7, 7), np.uint8)
kernels9 = np.ones((9, 9), np.uint8)

kernele3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernele5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernele7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
kernele9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

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


def readFromFile(filename):
    capSRC = f"videos/{filename}.mp4" # will need to be adjusted for each folder/ device
    return capSRC # note: where python read a file will start from the folder which you open it

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

def fillAndHealMask(binary_mask, min_area = 400, max_area = 60000, min_solidity = 0.55):
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

def filterCir(contours, image, min_circularity = 0.80,limit=1200): # độ tròn nên là 0.80 hoặc hơn
    for contour in contours:
        # Mặc kệ những contour nhỏ hơn 1200 pixels
        area = cv2.contourArea(contour)

        if area < limit:
            continue

        # Tính độ tròn của contour
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        circularity = (4 * np.pi * area) / (perimeter ** 2)

        if circularity < min_circularity:
            continue

        # Vẽ bounding box
        x, y, w, h = cv2.boundingRect(contour)

        if w < 25 or h < 25:
            continue

        # Filter by aspect ratio
        ratio = w / h

        if not 0.5 <= ratio <= 1.5:
            continue

        # Draw bounding box
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

def filterRec(contours, image, min_angle=75, max_angle=105, limit=2250):
    for contour in contours:

        area = cv2.contourArea(contour)

        if area < limit:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        # Xấp xỉ contour note: original is 0.04
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

        # Kiểm tra tỷ lệ chiều rộng / chiều cao
        ratio = w / h
        #      too tall        too wide
        if not 0.1 <= ratio <= 4.5:
            continue

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
        
def filterTri(contours, image, min_angle=25, max_angle=130, limit=1500):
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < limit:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        # Xấp xỉ contour
        epsilon = 0.02 * perimeter
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

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


def watchVideo(filename):
    capSRC = readFromFile(filename)
    
    cap = cv2.VideoCapture(capSRC)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # bottom blindfold
    blindfold = createEvilROI(height, width)
    
    if not cap.isOpened():
        raise ValueError("Cannot open video!")
    else:
        print("Video received successfully!")
        print(f"fps: {fps}")
        print(f"Height: {height}")
        print(f"Width: {width}")
        print("Notes:")
        print("Press SPACE or p to pause the video.")
        print("Press ESC or q to exit the video.")
        print("Press c to save the curent frame as a jpg.")

        
    # Tạo các cửa sổ nhỏ để có thể cùng lúc xem video và 2 masks
    cv2.namedWindow("Current Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Current Frame", int(width * 0.3), int(height * 0.3))
            
    paused = False
    while True:
        if not paused:

            ret, frame = cap.read()
            frame_clean = frame.copy()
            
            if not ret:
                break
                
            blur = cv2.GaussianBlur(frame, (5,5), 0)
            # holy hell! GaussianBlur, the GOAT!
        
            # lấy ảnh hsv
            blur_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
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
            # gre = getColorMask(blur_hsv,lower_green,upper_green)

            # pure mask list: red, gre, blu, yel        

            # apply blindfold
            red = red & blindfold
            yel = yel & blindfold
            blu = blu & blindfold
            # gre = cv2.bitwise_and(gre,blindfold)

            # remove stray pixels (MORPH_OPEN)
            red = removeStrays(red, kernele3, 2)
            yel = removeStrays(yel, kernele3, 2)
            blu = removeStrays(blu, kernele3, 2)
            # gre = removeStrays(gre, kernele3, 2)
            
            # attempt re connect (MORPH_CLOSE)
            red = mendCracks(red, kernele5, 3)
            yel = mendCracks(yel, kernele5, 3)
            blu = mendCracks(blu, kernele5, 3)
            # gre = mendCracks(gre, kernele7, 2)        
            
            # fill inside
            
            red = fillAndHealMask(red)
            yel = fillAndHealMask(yel)
            blu = fillAndHealMask(blu)
            # gre = fillAndHealMask(gre)        
            
            
            
            # pre_merge
            
            rb_collab = red | blu
            ry_collab = red | yel
            
            # post_merge
            

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
            filterRec(blu_contours, frame, min_angle=80, max_angle=110, limit=2000)
            filterCir(redblu_contours, frame, min_circularity=0.85, limit=1750)
            filterTri(redyel_contours, frame, min_angle=25, max_angle=130, limit=1750)
            # vẽ bounding boxes  
                
                
                
                
            # Hiển thị frame lên cửa sổ
            if frame is not None:
                cv2.imshow('Current Frame', frame)

        # wait key
        key = cv2.waitKey(1) & 0xFF

        # Nhấn phím Space (' ') hoặc phím 'p' để Tạm dừng / Tiếp tục
        if key == ord(' ') or key == ord('p'):
            paused = not paused
            
        if key == ord('c'):
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            outfilename = f"52400174_Output-{filename}-Frame-{frame_number}.jpg"

            success = cv2.imwrite("images/" + outfilename, frame_clean)

            if success:
                print(f"Saved: {outfilename}")
            else:
                print("Failed to save image!")
        
                
        # Nhấn phím 'q' hoặc ESC để thoát chương trình
        elif key == ord('q') or key == 27:
            break




# interface

while True:
    print("===============MENU===============")
    print("1. Watch video 1")
    print("2. Watch video 2")
    print("0. Quit")
    print("===============MENU===============")
    
    choice = input("Please pick an option: ").strip()
    match choice:
        case "0":
            break
        
        case "1":
            watchVideo(vid1)
            break
                
        case "2":
            watchVideo(vid2)
            break
                
        case _:
            print("==================================")
            print("Please input a valid choice.")
            print("==================================")