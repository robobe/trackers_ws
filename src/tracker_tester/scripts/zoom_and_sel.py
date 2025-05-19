import cv2

drawing = False
ix, iy = -1, -1
box = None
box_ready = False
frame = None
frame_f = None
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, box, box_ready, frame_f

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        box_ready = False
        roi = (x, y, 200, 200)
        frame_f = frame.copy()
        # x, y, w, h = roi
        # cropped = frame_f[max(0,y-h):y + h, max(0,x-w):x + w]
        # zoomed = cv2.resize(cropped, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)

        # frame_f = zoomed

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            frame_copy = frame_f.copy()
            cv2.rectangle(frame_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Live", frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        box = (min(ix, x), min(iy, y), abs(x - ix), abs(y - iy))
        box_ready = True
        frame_f = None
        print(f"Selected ROI: {box}")

cv2.namedWindow("Live")
cv2.setMouseCallback("Live", draw_rectangle)

cap = cv2.VideoCapture("/workspace/src/tracker_tester/data/tennis.mp4")  # Or use a path like 'video.mp4'

print("Click and drag to select ROI. Release mouse to confirm. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if box_ready and box:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if frame_f is not None:
        frame=frame_f
        print("ddd")
    cv2.imshow("Live", frame)
    key = cv2.waitKey(100) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
