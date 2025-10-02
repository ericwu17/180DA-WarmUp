import numpy as np
import cv2

cap = cv2.VideoCapture(0)

mouse_x, mouse_y = 0, 0

def mouse_callback(event, x, y, flags, param):
    """Callback function to track mouse position"""
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y

cv2.namedWindow('img')
cv2.setMouseCallback('img', mouse_callback)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    resized_image = cv2.resize(frame, (800, 500))

    hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
    if 0 <= mouse_y < hsv.shape[0] and 0 <= mouse_x < hsv.shape[1]:
        h, s, v = hsv[mouse_y, mouse_x]
        print(f"Position: ({mouse_x}, {mouse_y}) - HSV: ({h}, {s}, {v})", end='\r')

    cv2.imshow('img', resized_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()