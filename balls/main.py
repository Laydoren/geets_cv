import json
import random
from pathlib import Path

import cv2
import numpy as np

save_path = Path(__file__).parent
cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)

position = [0, 0]
clicked = False


def on_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        global position, clicked
        position = [x, y]
        clicked = True


cv2.setMouseCallback("Image", on_click)
capture = cv2.VideoCapture(0)

objects = []
config_path = save_path / "config.json"
if config_path.exists():
    with config_path.open("r") as f:
        js = json.load(f)
        for obj in js.get("objects", []):
            objects.append(
                (
                    np.array(obj["lower"], dtype="u1"),
                    np.array(obj["upper"], dtype="u1"),
                )
            )

secret = []
stable = 0
NEED = 20
win_timer = 0
game_mode = "row"


def new_secret():
    if game_mode == "row":
        if len(objects) >= 3:
            return random.sample(range(len(objects)), 3)
    else:
        if len(objects) >= 4:
            return random.sample(range(len(objects)), 4)
    return []


while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    blured = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blured, cv2.COLOR_BGR2HSV)
    key = cv2.waitKey(50) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("n") and (
        (game_mode == "row" and len(objects) >= 3)
        or (game_mode == "grid" and len(objects) >= 4)
    ):
        secret = new_secret()
        stable = 0
        win_timer = 0
    elif key == ord("m"):
        game_mode = "grid" if game_mode == "row" else "row"
        secret = new_secret()
        stable = 0
        win_timer = 0

    if clicked:
        clicked = False
        color = hsv[position[1], position[0]]
        new_lower = np.clip(color * 0.9, 0, 255).astype("u1")
        new_upper = np.clip(color * 1.1, 0, 255).astype("u1")
        new_upper[1] = 255
        new_upper[2] = 255
        objects.append((new_lower, new_upper))
        if not secret:
            secret = new_secret()

    ball_found = {}
    combined_mask = None

    for idx, (lower, upper) in enumerate(objects):
        inr = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(inr, cv2.MORPH_CLOSE, np.ones((5, 5), dtype="u1"))
        combined_mask = (
            mask if combined_mask is None else cv2.bitwise_or(combined_mask, mask)
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            (cx, cy), radius = cv2.minEnclosingCircle(c)
            if radius > 10:
                ball_found[idx] = (int(cx), int(cy), int(radius))

    if combined_mask is not None:
        cv2.imshow("Mask", combined_mask)

    for idx, (cx, cy, r) in ball_found.items():
        cv2.circle(frame, (cx, cy), r, (0, 255, 255), 4)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(
            frame,
            str(idx + 1),
            (cx - 10, cy - r - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    if secret:
        if game_mode == "row" and len(ball_found) >= 3:
            detected = [
                idx for idx, _ in sorted(ball_found.items(), key=lambda x: x[1][0])
            ][:3]

            if detected == secret:
                stable += 1
            else:
                stable = 0

        elif game_mode == "grid" and len(ball_found) >= 4:
            square_balls = list(ball_found.items())[:4]
            sort_y = sorted(square_balls, key=lambda x: x[1][1])

            top_row = sorted(sort_y[:2], key=lambda x: x[1][0])
            bottom_row = sorted(sort_y[2:], key=lambda x: x[1][0])

            detected = [
                top_row[0][0],
                top_row[1][0],
                bottom_row[0][0],
                bottom_row[1][0],
            ]

            if detected == secret:
                stable += 1
            else:
                stable = 0
        else:
            stable = 0

        if stable >= NEED:
            win_timer = 90
            stable = 0
            secret = new_secret()

    mode_text = "Mode: Grid" if game_mode == "grid" else "Mode: Row"
    cv2.putText(
        frame, mode_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
    )

    if secret:
        if game_mode == "row":
            task_text = "Sequence: " + " ".join(str(i + 1) for i in secret)
        else:
            task_text = f"Grid (TL TR BL BR): " + " ".join(str(i + 1) for i in secret)

        cv2.putText(
            frame,
            task_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    if win_timer > 0:
        win_timer -= 1
        cv2.putText(
            frame,
            "WIN!",
            (100, frame.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            (0, 255, 0),
            4,
        )

    cv2.imshow("Image", frame)

with (save_path / "config.json").open("w") as f:
    json.dump(
        {
            "objects": [
                {"lower": lo.tolist(), "upper": up.tolist()} for lo, up in objects
            ]
        },
        f,
    )
cv2.destroyAllWindows()
