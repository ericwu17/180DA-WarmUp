import numpy as np
import cv2

cap = cv2.VideoCapture(0)

USE_RGB_THRESHOLD = False

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


    if USE_RGB_THRESHOLD:
        lower_bound = np.array([70, 100, 230])
        upper_bound = np.array([120, 140, 256])
        mask = cv2.inRange(resized_image, lower_bound, upper_bound)
    else:
        hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
        if 0 <= mouse_y < hsv.shape[0] and 0 <= mouse_x < hsv.shape[1]:
            h, s, v = hsv[mouse_y, mouse_x]
            print(f"Position: ({mouse_x}, {mouse_y}) - HSV: ({h}, {s}, {v})", end='\r')

        lower_bound = np.array([2, 140, 210])
        upper_bound = np.array([10, 200, 256])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Remove noise using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)    
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)



    

    coords = cv2.findNonZero(cleaned)
    
    if coords is None:
        cv2.imshow('img', resized_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   
        continue
    
    # Get bounding rectangle from all non-zero pixels
    x, y, w, h = cv2.boundingRect(coords)
    
    # Draw bounding box
    cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imshow('img', resized_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()