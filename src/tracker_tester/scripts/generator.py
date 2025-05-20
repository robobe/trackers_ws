import cv2
import numpy as np
import math
import time

# Create a black background
width, height = 640, 480
canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Rectangle properties
rect_width, rect_height = 60, 40
amplitude = 100  # amplitude of sine wave
frequency = 0.01  # speed of sine wave

t = 0

while True:
    # Clear canvas
    frame = canvas.copy()

    # Calculate x and y positions
    x = int((width //2) + amplitude*math.sin(2 * math.pi * frequency * t))  # move steadily to the right
    y = int(height // 2 )

    # Draw rectangle
    cv2.rectangle(frame, (x, y), (x + rect_width, y + rect_height), (0, 255, 0), -1)

    # Show
    cv2.imshow("Sine Wave Rectangle", frame)

    t += 1
    key = cv2.waitKey(100) & 0xFF  # ~10 FPS
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()
