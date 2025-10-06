import cv2
import numpy as np
import math
import serial
import time

# Setup serial connection (change 'COM3' to your port if needed)
ser = serial.Serial('COM3', 115200, timeout=1)

SQ3 = math.sqrt(3)
kernel = np.ones((5,5),np.uint8)

def Nothing(x):
    pass    

cv2.namedWindow("Captured Frame")
cv2.createTrackbar("Kp", "Captured Frame", 0, 200, Nothing)
cv2.createTrackbar("Ki", "Captured Frame", 0, 200, Nothing)
cv2.createTrackbar("Kd", "Captured Frame", 0, 400, Nothing)

lower = np.array( [5, 90, 80])
upper = np.array([24, 255, 255]) 

def xy_to_axes3(x, y, center, motors, rB):
    axes_vals = []
    ball = np.array([x, y])
    for motor in motors:
        axis = motor - center
        axis = axis / np.linalg.norm(axis)
        val = np.dot(ball - center, axis)
        val_norm = val / rB
        axes_vals.append(val_norm)
    return axes_vals

def calibration(x,y,r):
    if r is None or r < 20 or r > 500:
        return x,y
    mm = 75/r
    return x*mm, y*mm

def centerle(pt01,pt10,pt02,pt20):
    if (pt01 == 0 and pt10 == 0) or (pt02 == 0 and pt20 == 0):
        return 0,0
    x = int(pt02) - int(pt01)
    y = int(pt10) - int(pt20)
    if abs(x) > 200 or abs(y) > 200:
        return 0,0
    return x, y

def blurImage(frame):
    return cv2.GaussianBlur(frame, (7,7), 1.5)

def TopDetect(frame, hsv):
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    result = cv2.bitwise_and(frame, frame, mask=mask)
    x = y = 0
    detected = False
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(c)
        x, y, r = int(x), int(y), int(r)
        if r > 15:
            detected = True
            cv2.circle(result, (x,y), r, (0,255,0), 2)
            cv2.circle(result, (x,y), 3, (0,0,255), -1)
    return result, x, y, detected

def backgroundDetect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = blurImage(gray)
    inv = cv2.bitwise_not(gray)
    _, th = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    img_blur = blurImage(th)
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=50, maxRadius=200)
    x = y = r = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = map(int, circles[0,0])
        cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 7, (0, 0, 255), -1)
    return frame, x, y, r

def draw_axes(frame, center, motors, rB):
    colors = [(255,0,0), (0,255,0), (0,0,255)]
    for i, motor in enumerate(motors):
        cv2.line(frame, tuple(center.astype(int)), tuple(motor.astype(int)), colors[i], 2)
        cv2.circle(frame, tuple(motor.astype(int)), 8, colors[i], -1)
        cv2.putText(frame, f"M{i+1}", tuple((motor+10).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)

cap = cv2.VideoCapture(0)  # webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frameHSV = blurImage(hsv)

    Kp = cv2.getTrackbarPos("Kp", "Captured Frame")
    Ki = cv2.getTrackbarPos("Ki", "Captured Frame")
    Kd = cv2.getTrackbarPos("Kd", "Captured Frame")

    topDetectFrame, xT ,yT, ball_detected = TopDetect(frame, hsv)
    backgroundDetectFrame, xB, yB, rB = backgroundDetect(frame)

    if rB != 0 and ball_detected:
        center = np.array([xB, yB])
        motors = [
            center + np.array([rB, 0]),                    # Motor 1: right
            center + np.array([-rB/2, rB*SQ3/2]),          # Motor 2: top left
            center + np.array([-rB/2, -rB*SQ3/2])          # Motor 3: bottom left
        ]
        draw_axes(backgroundDetectFrame, center, motors, rB)

        ball = np.array([xT, yT])
        cv2.circle(backgroundDetectFrame, tuple(ball.astype(int)), 8, (0,255,255), -1)

        axes_vals = xy_to_axes3(xT, yT, center, motors, rB)
        axes_int = [int(round(val * 80)) for val in axes_vals]  # scale and round to int

        # Display integer axis values on frame
        for i, val in enumerate(axes_int):
            txt = f"A{i+1}: {val}"
            cv2.putText(backgroundDetectFrame, txt, (20, 40 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        msg = "{},{},{},{},{},{}\n".format(axes_int[0], axes_int[1], axes_int[2], Kp, Ki, Kd)
    else:
        msg = "0,0,0,{},{},{}\n".format(Kp, Ki, Kd)

    print(msg.strip())
    ser.write(msg.encode())
    # line = ser.readline().decode().strip()
    # if line:
    #     print("STM32 replied:", line)
    # time.sleep(0.01)  

    cv2.imshow("Captured Frame", backgroundDetectFrame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()