import numpy as np
import cv2

# video links
vid1 = "video1"
vid2 = "video2"
vid3 = "video1_output_sample"

# kernels
kernel3 = np.ones((3, 3), np.uint8)
kernel5 = np.ones((5, 5), np.uint8)


def readFromFile(filename):
    capSRC = f"videos/{filename}.mp4"
    return capSRC

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
        [int(width * 0.35), int(height * 0.52)], # Điểm 2: Góc trên bên trái (ngay dưới đường chân trời)
        [int(width * 0.65), int(height * 0.52)], # Điểm 3: Góc trên bên phải (ngay dưới đường chân trời)
        [int(width * 0.99), height]           # Điểm 4: Góc dưới bên phải (gần mép phải)
    ], dtype=np.int32)
    
    # Tô đen (0) cho vùng bên trong hình thang
    cv2.fillPoly(trapezoidMask, [pts], 0)
    
    mask = cv2.bitwise_and(trapezoidMask, rectangleMask)
    
    return mask

capSRC = readFromFile(vid1)

cap = cv2.VideoCapture(capSRC)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

# tạo mask che 1/3 màn hình bên dưới để giảm nhiễu
# blindfold = np.ones((height, width), dtype=np.uint8) * 255
# blindfold[int(height * 0.667):, :] = 0 
# tạo mask che 1/3 màn hình bên dưới để giảm nhiễu
    
# Tạo các cửa sổ nhỏ để có thể cùng lúc xem video và 4 masks
cv2.namedWindow("Current Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Current Frame", int(width * 0.22), int(height * 0.22))
cv2.namedWindow("red_mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("red_mask", int(width * 0.22), int(height * 0.22))
cv2.namedWindow("yellow_mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("yellow_mask", int(width * 0.22), int(height * 0.22))    
cv2.namedWindow("blue_mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("blue_mask", int(width * 0.22), int(height * 0.22))
cv2.namedWindow("green_mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("green_mask", int(width * 0.22), int(height * 0.22))
cv2.namedWindow("temp_screen_1", cv2.WINDOW_NORMAL)
cv2.resizeWindow("temp_screen_1", int(width * 0.22), int(height * 0.22))
cv2.namedWindow("temp_screen_2", cv2.WINDOW_NORMAL)
cv2.resizeWindow("temp_screen_2", int(width * 0.22), int(height * 0.22))






# các đoạn màu

# đỏ
# đỏ 1
lower_red1 = np.array([0, 60, 40]) # personally, np.array([0, 60, 40])
upper_red1 = np.array([5, 255, 255]) #             np.array([5, 255, 255])
# đỏ 2
lower_red2 = np.array([160, 60, 40]) # personally, np.array([160, 60, 40])
upper_red2 = np.array([180, 255, 255]) #             np.array([180, 255, 255])

# vàng
lower_yellow = np.array([15, 100, 100])
upper_yellow = np.array([35, 255, 255])

# xanh dương
lower_blue = np.array([105, 140, 70]) # personally, np.array([105, 120, 45]),
upper_blue = np.array([125, 255, 255]) #            np.array([115, 255, 255])


# xanh lá
lower_green = np.array([35, 80, 60])
upper_green = np.array([85, 255, 255])

# bottom blindfold (1/3)
blindfold = createEvilROI(height, width)



paused = False
while True:
    if not paused:
        
        ret, frame = cap.read()
        
        if not ret:
            break
            
        blur = cv2.GaussianBlur(frame, (5,5), 0)
            
            
        
        
        
        
        # lấy ảnh hsv
        blur_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # lấy ảnh hsv    
        
        # mask đỏ
        red1 = cv2.inRange(
            blur_hsv,
            lower_red1,
            upper_red1
        )
        red2 = cv2.inRange(
            blur_hsv,
            lower_red2,
            upper_red2
        )
        red = cv2.bitwise_or(red1, red2)

        # mask vàng
        yel = cv2.inRange(
            blur_hsv,
            lower_yellow,
            upper_yellow
        )
        # mask xanh dương
        blu = cv2.inRange(
            blur_hsv,
            lower_blue,
            upper_blue
        )
        
        # mask xanh lá
        gre = cv2.inRange(
            blur_hsv,
            lower_green,
            upper_green
        )
        
        
        # apply blindfold
        red = cv2.bitwise_and(red,blindfold)
        blu = cv2.bitwise_and(blu,blindfold)
        yel = cv2.bitwise_and(yel,blindfold)
        gre = cv2.bitwise_and(gre,blindfold)
        
        
        
        
        if frame is not None:
            cv2.imshow('Current Frame', frame)
            cv2.imshow('red_mask', red)
            cv2.imshow('yellow_mask', yel)                
            cv2.imshow('blue_mask', blu)
            cv2.imshow('green_mask', gre)
            cv2.imshow('temp_screen_1',blindfold)



        
    # wait key
    key = cv2.waitKey(1) & 0xFF

    # Nhấn phím Space (' ') hoặc phím 'p' để Tạm dừng / Tiếp tục
    if key == ord(' ') or key == ord('p'):
        paused = not paused
        
    if key == ord('c'):
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        outfilename = f"images/52400174_Output-{vid1}-Frame-{frame_number}.jpg"

        success = cv2.imwrite(outfilename, frame)

        if success:
            print(f"Saved: {outfilename}")
        else:
            print("Failed to save image!")
    
            
    # Nhấn phím 'q' hoặc ESC để thoát chương trình
    elif key == ord('q') or key == 27:
        break

