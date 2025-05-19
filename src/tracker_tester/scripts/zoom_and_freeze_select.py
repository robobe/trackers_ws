import cv2

all_break = False
freeze_image = None
allow_zoom = False
sx, sy, ex, ey, drawing, box, box_ready = -1, -1, -1, -1, False, None, False

def draw_rectangle(event, x, y, flags, param):
    global sx, sy, ex, ey, drawing, box, box_ready

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        sx, sy = x, y
        box_ready = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            ex, ey = x, y
            

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        box = (min(sx, x), min(sy, y), abs(x - sx), abs(y - sy))
        box_ready = True
        print(f"Selected ROI: {box}")

cv2.namedWindow("Live")
cv2.setMouseCallback("Live", draw_rectangle)

while True:
    cap = cv2.VideoCapture("/workspace/src/tracker_tester/data/tennis.mp4")  # Or use a path like 'video.mp4'

    zoom_factor = 2  # 2x zoom
    

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        original_frame = frame.copy()

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
            frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        if drawing:
            if freeze_image is not None:
                if allow_zoom:
                    frame = frame.copy()
                else:
                    frame = freeze_image.copy()
            cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)

        if box_ready and box:
            x, y, w, h = box
            if allow_zoom:
                scale_x = 0.5 #(x2 - x1) / w
                scale_y = 0.5 #(y2 - y1) / h
                x = int(x1 + x * scale_x)
                y = int(y1 + y * scale_y)
                w = int(w * scale_x)
                h = int(h * scale_y)
                allow_zoom = False
                box = x,y,w,h
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
        cv2.imshow("Live", frame)

        key = cv2.waitKey(100) & 0xFF
        if key == 27:  # ESC
            all_break = True
            break

        if key == ord("c"):  # zoom
            box_ready = False
            box = None

        if key == ord("z"):  # zoom
            allow_zoom = not allow_zoom
            print("zoom: {}".format(allow_zoom))

        if key == ord("f"):  # freeze
            print("Freeze")
            if freeze_image is None:
                freeze_image = original_frame.copy()
            else:
                freeze_image = None
                print("Unfreeze")

    cap.release()

    if all_break:
        break
    
cv2.destroyAllWindows()
