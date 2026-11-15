
import cv2
import numpy as np

vid1 = "video1"
vid2 = "video2"


def readFromFile(filename):
    capSRC = f"Digital_Image_Processing/videos/{filename}.mp4" # will need to be adjusted for each folder/ device
    return capSRC # note: where python read a file will start from the folder which you open it

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

        
    # Tạo các cửa sổ nhỏ để có thể cùng lúc xem video và 2 masks
    cv2.namedWindow("Current Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Current Frame", int(width * 0.3), int(height * 0.3))
            
    paused = False
    while True:
        if not paused:

            ret, frame = cap.read()
            
            if not ret:
                break
                
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

            success = cv2.imwrite("Digital_Image_Processing/images/" + outfilename, frame)

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