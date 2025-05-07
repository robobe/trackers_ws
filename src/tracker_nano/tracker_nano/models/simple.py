"""
Demo with model v2
"""
import cv2
import pathlib
import sys


VIDEO_FILE = "vtest.avi"
 
def preprocess_frame(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def create_tracker():
    # Create tracker object
    params = cv2.TrackerNano.Params()
    # v2
    params.backbone = "/workspace/models/nanotrack_backbone_sim.onnx"
    params.neckhead = "/workspace/models/nanotrack_head_sim.onnx"

    
    params.backend = cv2.dnn.DNN_BACKEND_DEFAULT
    params.target  = cv2.dnn.DNN_TARGET_CPU

    tracker = cv2.TrackerNano.create(params)  # CSRT tracker offers good accuracy
    return tracker

def initialize_tracker(frame, tracker):
    # Select ROI for tracking
    bbox = cv2.selectROI('Tracking', frame, fromCenter=False, showCrosshair=True)
    if bbox[2] <= 0 or bbox[3] <= 0:
        sys.exit("ROI selection cancelled. Exiting...")
    
    # Initialize tracker
    try:
        tracker.init(frame, bbox)
    except Exception as e:
        print('Unable to initialize tracker with requested bounding box. Is there any object?')
        print(e)
        print('Try again ...')
        sys.exit(1)
    return bbox

def main():
    video_path = pathlib.Path(__file__).parent.joinpath("..").joinpath("data").joinpath(VIDEO_FILE)
    print(video_path.as_posix())
    video_path = video_path.as_posix()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        exit()

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Cannot read video file")
        sys.exit()

    # Create tracker and initialize
    tracker = create_tracker()

    frame = preprocess_frame(frame)
    bbox = initialize_tracker(frame, tracker)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Update tracker
        frame = preprocess_frame(frame)
        ok, bbox = tracker.update(frame)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        # Display the frame
        cv2.imshow('nano', frame)

        # Press 'q' to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()