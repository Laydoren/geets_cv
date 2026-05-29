import time
import cv2
import mss
import numpy as np
import pyautogui

with mss.mss() as sct:
    screenshot = np.array(sct.grab(sct.monitors[1]))
    screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    roi = cv2.selectROI("Select Area", screenshot_bgr, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Area")

    SCAN_AREA = {"top": int(roi[1]), "left": int(roi[0]), "width": int(roi[2]), "height": int(roi[3])}

if SCAN_AREA["width"] == 0 or SCAN_AREA["height"] == 0:
    print("ROI wasnt chosen")
    exit()

time.sleep(3)
pyautogui.press("space")
start_time = time.time()

with mss.mss() as sct:
    while True:
        img = np.array(sct.grab(SCAN_AREA))
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        _, mask = cv2.threshold(gray, 149, 255, cv2.THRESH_BINARY_INV)
        yc, xc = np.where(mask > 0)

        if len(xc) > 0:
            leading_idx = np.argmin(xc)
            closest_x, closest_y = (xc[leading_idx], yc[leading_idx])

            jump_threshold = min(38  + int((time.time() - start_time) * 1.5), 310)
            if closest_x < jump_threshold:
                if closest_y < (SCAN_AREA["height"] * 0.8):
                    pyautogui.keyDown("down")
                    time.sleep(0.18)
                    pyautogui.keyUp("down")

                else:
                    pyautogui.press("space")
                    time.sleep(0.11)
                    pyautogui.keyDown("down")
                    time.sleep(0.033)
                    pyautogui.keyUp("down")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()