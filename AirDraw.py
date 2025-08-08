import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Create a white canvas
canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255

# Color palette (BGR)
colors = [(0,0,0), (0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0)]
color_names = ['Black', 'Red', 'Green', 'Blue', 'Yellow', 'Magenta', 'Cyan']
current_color = 0

# Exit and Clear button coordinates
exit_btn_pos = (540, 10, 630, 50)   # (x1, y1, x2, y2)
clear_btn_pos = (420, 10, 510, 50)  # (x1, y1, x2, y2)

def draw_palette(img, selected):
    for i, color in enumerate(colors):
        x1, y1 = 10 + i*50, 10
        x2, y2 = x1 + 40, y1 + 40
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        if i == selected:
            cv2.rectangle(img, (x1, y1), (x2, y2), (128,128,128), 3)
        cv2.putText(img, color_names[i], (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,50,50), 1)
    # Draw Clear button
    x1, y1, x2, y2 = clear_btn_pos
    cv2.rectangle(img, (x1, y1), (x2, y2), (200,200,200), -1)
    cv2.putText(img, "CLEAR", (x1+10, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    # Draw Exit button
    x1, y1, x2, y2 = exit_btn_pos
    cv2.rectangle(img, (x1, y1), (x2, y2), (50,50,50), -1)
    cv2.putText(img, "EXIT", (x1+20, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

# Start webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Drawing Canvas")

prev_x, prev_y = None, None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for natural interaction
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Draw palette, clear, and exit button on both canvas and frame
    draw_palette(canvas, current_color)
    draw_palette(frame, current_color)

    exit_by_finger = False
    clear_by_finger = False

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get index finger tip coordinates
            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
            y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

            # Draw only when index finger is up and middle finger is down
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

            index_up = index_tip.y < index_pip.y
            middle_up = middle_tip.y < middle_pip.y

            # Exit if index finger tip touches the exit button
            x1, y1, x2, y2 = exit_btn_pos
            if x1 <= x <= x2 and y1 <= y <= y2 and index_up:
                exit_by_finger = True
                break

            # Clear if index finger tip touches the clear button
            x1, y1, x2, y2 = clear_btn_pos
            if x1 <= x <= x2 and y1 <= y <= y2 and index_up:
                clear_by_finger = True
                break

            # Color palette selection
            if index_up and not middle_up and y < 60:
                for i in range(len(colors)):
                    x1, y1 = 10 + i*50, 10
                    x2, y2 = x1 + 40, y1 + 40
                    if x1 < x < x2 and y1 < y < y2:
                        current_color = i
                        prev_x, prev_y = None, None
                        break

            # Drawing
            elif index_up and not middle_up and y >= 60:
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), colors[current_color], 5)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = None, None

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Clear the canvas if clear button is touched
    if clear_by_finger:
        canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
        prev_x, prev_y = None, None

    # Show the canvas and camera feed
    cv2.imshow("Drawing Canvas", canvas)
    cv2.imshow("Camera Feed", frame)

    # Exit if 'q' is pressed, window is closed, or exit button is touched by finger
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("Drawing Canvas", cv2.WND_PROP_VISIBLE) < 1 or \
       cv2.getWindowProperty("Camera Feed", cv2.WND_PROP_VISIBLE) < 1:
        break
    if exit_by_finger:
        break

cap.release()
cv2.destroyAllWindows()