import cv2
import mediapipe as mp
import numpy as np
import time
import os

# -----------------------------
# AirDraw AI - Gesture Drawing
# -----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opening. Please check your webcam permission.")
    exit()

cap.set(3, 1280)
cap.set(4, 720)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

canvas = None

draw_color = (255, 0, 255)   # Pink
mode = "DRAW"

prev_x, prev_y = 0, 0
last_click_time = 0

# Button format: name, x1, y1, x2, y2, color
buttons = [
    ("PINK", 20, 15, 140, 65, (255, 0, 255)),
    ("BLUE", 160, 15, 280, 65, (255, 0, 0)),
    ("GREEN", 300, 15, 420, 65, (0, 255, 0)),
    ("ERASER", 440, 15, 580, 65, (0, 0, 0)),
    ("CLEAR", 600, 15, 740, 65, (80, 80, 80)),
    ("SAVE", 760, 15, 900, 65, (0, 180, 255)),
]


def draw_toolbar(img):
    overlay = img.copy()

    cv2.rectangle(overlay, (0, 0), (1280, 85), (20, 20, 20), -1)
    img[:] = cv2.addWeighted(overlay, 0.65, img, 0.35, 0)

    for name, x1, y1, x2, y2, color in buttons:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

        text_color = (255, 255, 255)
        if name == "GREEN":
            text_color = (0, 0, 0)

        cv2.putText(
            img,
            name,
            (x1 + 18, y1 + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            text_color,
            2
        )

    cv2.putText(
        img,
        f"Mode: {mode}",
        (950, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )


def is_inside_button(x, y):
    for name, x1, y1, x2, y2, color in buttons:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return name, color
    return None, None


def save_drawing():
    if canvas is None:
        return

    folder = "saved_drawings"
    os.makedirs(folder, exist_ok=True)

    filename = f"{folder}/air_drawing_{int(time.time())}.png"

    # Black background + drawing
    black_bg = np.zeros_like(canvas)
    final_img = cv2.addWeighted(black_bg, 1, canvas, 1, 0)

    cv2.imwrite(filename, final_img)
    print(f"Saved: {filename}")


def draw_glow_line(img, start, end, color):
    # Glow effect
    cv2.line(img, start, end, color, 22)
    cv2.line(img, start, end, color, 14)
    cv2.line(img, start, end, (255, 255, 255), 4)


while True:
    success, frame = cap.read()

    if not success:
        print("Camera frame not received.")
        break

    frame = cv2.flip(frame, 1)

    h, w, c = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    draw_toolbar(frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark

            index_tip = lm[8]
            index_pip = lm[6]
            middle_tip = lm[12]
            middle_pip = lm[10]

            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            index_up = index_tip.y < index_pip.y
            middle_up = middle_tip.y < middle_pip.y

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Finger pointer circle
            cv2.circle(frame, (x, y), 12, draw_color, -1)
            cv2.circle(frame, (x, y), 22, draw_color, 2)

            selected_button, selected_color = is_inside_button(x, y)

            current_time = time.time()

            if selected_button and current_time - last_click_time > 0.7:
                if selected_button == "CLEAR":
                    canvas = np.zeros((h, w, 3), dtype=np.uint8)
                    prev_x, prev_y = 0, 0

                elif selected_button == "SAVE":
                    save_drawing()

                elif selected_button == "ERASER":
                    mode = "ERASER"
                    draw_color = (0, 0, 0)

                else:
                    mode = "DRAW"
                    draw_color = selected_color

                last_click_time = current_time

            # Drawing rule:
            # Only index finger up = draw
            # Index + middle finger up = move without drawing
            if y > 90:
                if index_up and not middle_up:
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = x, y

                    if mode == "DRAW":
                        draw_glow_line(canvas, (prev_x, prev_y), (x, y), draw_color)

                    elif mode == "ERASER":
                        cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), 45)

                    prev_x, prev_y = x, y

                else:
                    prev_x, prev_y = 0, 0

    # Combine camera and drawing canvas
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)

    output = cv2.add(frame_bg, canvas_fg)

    cv2.putText(
        output,
        "AirDraw AI | Index finger = Draw | Two fingers = Move | Q = Quit",
        (20, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.imshow("AirDraw AI", output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("c"):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    if key == ord("s"):
        save_drawing()

cap.release()
cv2.destroyAllWindows()