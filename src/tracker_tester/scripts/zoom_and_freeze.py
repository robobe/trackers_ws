import cv2

all_break = False
freeze_image = None
allow_zoom = False

while True:
    cap = cv2.VideoCapture("/workspace/src/tracker_tester/data/tennis.mp4")  # Or use a path like 'video.mp4'

    zoom_factor = 2  # 2x zoom
    

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if freeze_image is not None:
            frame = freeze_image

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
            zoom_frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        if allow_zoom:
            cv2.imshow("Live", zoom_frame)
        else:
            cv2.imshow("Live", frame)

        key = cv2.waitKey(100) & 0xFF
        if key == 27:  # ESC
            all_break = True
            break

        if key == ord("z"):  # zoom
            allow_zoom = not allow_zoom
            print("zoom: {}".format(allow_zoom))

        if key == ord("f"):  # freeze
            print("Freeze")
            if freeze_image is None:
                freeze_image = frame.copy()
            else:
                freeze_image = None
                print("Unfreeze")

    cap.release()

    if all_break:
        break
    
cv2.destroyAllWindows()
