import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def air_draw(frame, hands, canvas, prev_point):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    curr_point = prev_point

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)
            curr_point = (x, y)

            # Draw a black line from previous point to current point
            if prev_point is not None:
                cv2.line(canvas, prev_point, curr_point, (0, 0, 0), 5)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        curr_point = None

    return canvas, curr_point