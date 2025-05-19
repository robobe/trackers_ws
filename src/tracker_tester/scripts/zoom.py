import cv2

all_break = False

while True:
    cap = cv2.VideoCapture("/workspace/src/tracker_tester/data/tennis.mp4")  # Or use a path like 'video.mp4'

    zoom_factor = 2  # 2x zoom
    allow_zoom = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        if allow_zoom:
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2  # zoom center

            # Calculate cropped region
            crop_w = w // zoom_factor
            crop_h = h // zoom_factor
            x1 = max(cx - crop_w // 2, 0)
            y1 = max(cy - crop_h // 2, 0)
            x2 = min(cx + crop_w // 2, w)
            y2 = min(cy + crop_h // 2, h)

            cropped = frame[y1:y2, x1:x2]
            frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        cv2.imshow("Live", frame)
        key = cv2.waitKey(100) & 0xFF
        if key == 27:  # ESC
            all_break = True
            break

        if key == ord("z"):  # ESC
            allow_zoom = not allow_zoom

    cap.release()

    if all_break:
        break
    
cv2.destroyAllWindows()
