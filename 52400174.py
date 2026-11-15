import numpy as np
import cv2

# video links
vid1 = "video1"
vid2 = "video2"
vid3 = "video1_output_sample"

# kernels
kernel3 = np.ones((3, 3), np.uint8)
kernel5 = np.ones((5, 5), np.uint8)
kernel7 = np.ones((7, 7), np.uint8)
kernele5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernelredconnect = np.array([
    [0,1,0],
    [0,1,0],
    [0,1,0],
    [0,1,0],
    [0,1,0],
], dtype=np.uint8)
kernelblueconnect = np.array([
    [0,0,0,0,0,0,1],
    [0,0,0,0,0,1,0],
    [0,0,0,0,1,0,0],
    [0,0,0,1,0,0,0],
    [0,0,1,0,0,0,0],
    [0,1,0,0,0,0,0],
    [1,0,0,0,0,0,0],
], dtype=np.uint8)

# functions
def readFromFile(filename):
    capSRC = f"{filename}.mp4"
    return capSRC
    
def medianFilter_Iteration(*masks, iterations=1, s=3):
    processed_masks = []
    
    for m in masks:
        for _ in range(iterations):
            m = cv2.medianBlur(m, ksize=s)
        processed_masks.append(m)
        
    return processed_masks
    
def filter_cir_draw(contours, image, min_circularity = 0.75,limit=1750): # độ tròn nên là 0.7 hoặc hơn
    for contour in contours:
        # Mặc kệ những contour nhỏ hơn 2000 pixels
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

        if w < 15 or h < 15:
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
            
def processVideo(filename):
    capSRC = readFromFile(filename)
    
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
        print("Processing video...")
    
    # parameters của text
    text = "52400174"
    org = (width - 250, height - 50)             
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.2           
    color = (0, 255, 0)        
    thickness = 2              
    lineType = cv2.LINE_AA     
    
    # tạo mask che 1/3 màn hình bên dưới để giảm nhiễu
    blindfold = np.ones((height, width), dtype=np.uint8) * 255
    blindfold[int(height * 0.667):, :] = 0    

    # Set up output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        f"52400174_{filename}_output.mp4",
        fourcc,
        fps,
        (width, height)
    )
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
    
        # lấy ảnh hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # lấy ảnh hsv    
        
        
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
    
        # connect potentially disconnected signs
        red_mask = cv2.dilate(red_mask,kernel=kernelredconnect)
        blue_mask = cv2.dilate(blue_mask,kernel=kernelblueconnect)
        # connect potentially disconnected signs    
    
    
        # adjusting masks
        # remove stray pixels
        # red_mask = cv.medianBlur(red_mask,ksize=3) original
        # blue_mask = cv.medianBlur(blue_mask,ksize=3)
        red_mask, blue_mask = medianFilter_Iteration(red_mask, blue_mask, iterations=4,s=3)
        # remove stray pixels

        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_CLOSE,
            kernel5
        )

        blue_mask = cv2.morphologyEx(
            blue_mask,
            cv2.MORPH_CLOSE,
            kernel5
        )

        # dilate masks
        red_mask = cv2.dilate(red_mask,kernele5,iterations=1)
        blue_mask = cv2.dilate(blue_mask,kernele5,iterations=1)
        # dilate masks
        # adjusting masks
        # get masks
    
        # đưa blindfold lên che hết 1/3 dưới của màn hình
        red_mask = cv2.bitwise_and(red_mask,blindfold)
        blue_mask = cv2.bitwise_and(blue_mask,blindfold)
        # đưa blindfold lên che hết 1/3 dưới của màn hình
    
        # vẽ contour cho mask đỏ
        red_contours, _ = cv2.findContours(
            red_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        # vẽ contour cho mask đỏ

        # vẽ contour cho mask xanh
        blue_contours, _ = cv2.findContours(
            blue_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )        
        # vẽ contour cho mask xanh


        # vẽ bounding boxes
        filter_cir_draw(red_contours, frame,min_circularity=0.77,limit=1750)
        filter_cir_draw(blue_contours, frame,min_circularity=0.77,limit=1750)        
        # vẽ bounding boxes   
        
        # ghi MSSV vào
        cv2.putText(frame, text, org, fontFace, fontScale, color, thickness, lineType)
        
        if frame is not None:
            out.write(frame)
    
    cap.release()
    out.release()
    print(f"{filename} processed.")

def watchVideo(filename):
    capSRC = readFromFile(filename)
    
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
    blindfold = np.ones((height, width), dtype=np.uint8) * 255
    blindfold[int(height * 0.667):, :] = 0 
    # tạo mask che 1/3 màn hình bên dưới để giảm nhiễu
        
    # Tạo các cửa sổ nhỏ để có thể cùng lúc xem video và 2 masks
    cv2.namedWindow("Current Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Current Frame", int(width * 0.3), int(height * 0.3))
    cv2.namedWindow("red_mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("red_mask", int(width * 0.3), int(height * 0.3))
    cv2.namedWindow("blue_mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("blue_mask", int(width * 0.3), int(height * 0.3))    
    
    # parameters của text
    text = "52400174"
    org = (width - 250, height - 50)             
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.2           
    color = (0, 255, 0)        
    thickness = 4              
    lineType = cv2.LINE_AA  
        
    paused = False
    while True:
        if not paused:

            ret, frame = cap.read()
            
            if not ret:
                break
        
            # lấy ảnh hsv
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # lấy ảnh hsv    
            
            
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
        
            # connect potentially disconnected signs
            red_mask = cv2.dilate(red_mask,kernel=kernelredconnect)
            blue_mask = cv2.dilate(blue_mask,kernel=kernelblueconnect)
            # connect potentially disconnected signs    
        
        
            # adjusting masks
            # remove stray pixels
            red_mask = cv2.medianBlur(red_mask,ksize=3) 
            blue_mask = cv2.medianBlur(blue_mask,ksize=3)
            red_mask, blue_mask = medianFilter_Iteration(red_mask, blue_mask, iterations=4,s=3)
            # remove stray pixels

            red_mask = cv2.morphologyEx(
                red_mask,
                cv2.MORPH_CLOSE,
                kernel5
            )

            blue_mask = cv2.morphologyEx(
                blue_mask,
                cv2.MORPH_CLOSE,
                kernel5
            )

            # dilate masks
            red_mask = cv2.dilate(red_mask,kernele5,iterations=1)
            blue_mask = cv2.dilate(blue_mask,kernele5,iterations=1)
            # dilate masks
            # adjusting masks
            # get masks
        
            # đưa blindfold lên che hết 1/3 dưới của màn hình
            red_mask = cv2.bitwise_and(red_mask,blindfold)
            blue_mask = cv2.bitwise_and(blue_mask,blindfold)
            # đưa blindfold lên che hết 1/3 dưới của màn hình        
        
        
            # vẽ contour cho mask đỏ
            red_contours, _ = cv2.findContours(
                red_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            # vẽ contour cho mask đỏ

            # vẽ contour cho mask xanh
            blue_contours, _ = cv2.findContours(
                blue_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )        
            # vẽ contour cho mask xanh

            # vẽ bounding boxes
            filter_cir_draw(red_contours, frame,min_circularity=0.77,limit=1750)
            filter_cir_draw(blue_contours, frame,min_circularity=0.77,limit=1750)        
            # vẽ bounding boxes   
            
            # ghi MSSV vào
            cv2.putText(frame, text, org, fontFace, fontScale, color, thickness, lineType)
            
            # Hiển thị frame lên cửa sổ
            if frame is not None:
                cv2.imshow('Current Frame', frame)
                cv2.imshow('red_mask', red_mask)
                cv2.imshow('blue_mask', blue_mask)

        # wait key
        key = cv2.waitKey(1) & 0xFF

        # Nhấn phím Space (' ') hoặc phím 'p' để Tạm dừng / Tiếp tục
        if key == ord(' ') or key == ord('p'):
            paused = not paused
            
        if key == ord('c'):
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            outfilename = f"52400174_Output-{filename}-Frame-{frame_number}.jpg"

            success = cv2.imwrite(outfilename, frame)

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
            

            
