import cv2

# Giới hạn kích thước tối đa của cửa sổ hiển thị (chỉnh lại theo màn hình của bạn)
MAX_WIDTH = 1280
MAX_HEIGHT = 720


def on_mouse_click(event, x, y, flags, param):
  if event == cv2.EVENT_LBUTTONDOWN:
    h, s, v = hsv_img[y, x]
    print(f"Tọa độ gốc (x, y): ({x}, {y}) | Giá trị HSV: ({h}, {s}, {v})")

    img_display = img.copy()
    text = f"({x}, {y}) HSV: ({h}, {s}, {v})"

    # Tự động phóng to nét vẽ và chữ nếu ảnh gốc quá lớn để dễ nhìn khi thu nhỏ cửa sổ
    radius = max(4, int(5 / scale))
    font_scale = max(0.5, 0.6 / scale)
    thickness = max(1, int(1.5 / scale))

    cv2.circle(img_display, (x, y), radius, (0, 255, 0), -1)
    cv2.putText(
        img_display,
        text,
        (x + radius * 2, y - radius),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )
    cv2.imshow("Image", img_display)


img = cv2.imread("images/52400174_Output-video1-Frame-553.jpg")

if img is None:
  print("Không thể mở ảnh. Hãy kiểm tra lại đường dẫn file.")
else:
  hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  # 1. Tạo cửa sổ với cờ WINDOW_NORMAL (cho phép thay đổi kích thước)
  cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

  # 2. Tính tỷ lệ thu nhỏ để vừa khung hình nhưng không làm méo ảnh (tối đa = 1.0 để không phóng to ảnh nhỏ)
  h_img, w_img = img.shape[:2]
  scale = min(MAX_WIDTH / w_img, MAX_HEIGHT / h_img, 1.0)
  new_w = int(w_img * scale)
  new_h = int(h_img * scale)

  # 3. Ép kích thước cửa sổ theo kích thước đã tính
  cv2.resizeWindow("Image", new_w, new_h)

  cv2.setMouseCallback("Image", on_mouse_click)
  cv2.imshow("Image", img)

  cv2.waitKey(0)
  cv2.destroyAllWindows()