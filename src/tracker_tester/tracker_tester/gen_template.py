#!/bin/sh

import onnxruntime as ort
import cv2



WIN_NAME = "nanotrack"
drawing = False
start_point = (-1, -1)
end_point = (-1, -1)
boxes = []
frame = None

def on_mouse(event, x, y, flags, param):
    global start_point, end_point, drawing, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    # elif event == cv2.EVENT_MOUSEMOVE:
    #     if drawing:
    #         end_point = (x, y)
    #         # img = frame.copy()
    #         cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        # cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
        x1, y1 = start_point
        x2, y2 = end_point
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        print(f"Selected BBox: x={x_min}, y={y_min}, w={x_max - x_min}, h={y_max - y_min}")
        # Sort coordinates
        x1, y1 = start_point
        x2, y2 = end_point
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        # Crop and save the selected area
        cropped = frame[y_min:y_max, x_min:x_max]
        filename = f"/workspace/src/tracker_tester/data/template.jpg"
        cv2.imwrite(filename, cropped)


cv2.namedWindow(WIN_NAME)
cv2.setMouseCallback(WIN_NAME, on_mouse, 0)

def play_mp4_video(video_path):
    global frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    ret, frame = cap.read()

    
    # cv2.imshow(WIN_NAME, frame)
    cv2.imwrite("/workspace/src/tracker_tester/data/search.jpg", frame)
        

    cap.release()
    cv2.destroyAllWindows()

# Example usage:
play_mp4_video('/workspace/src/tracker_tester/data/tennis.mp4')