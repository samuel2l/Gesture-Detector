import os
import pickle #This module serializes and saves Python objects to a file, making it easier to load the data later.
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# mp.solutions.hands:initializes the hand tracking solution from MediaPipe, which will detect hand landmarks in images.
# mp_drawing:helps draw the detected hand landmarks on the image
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
# static_image_mode=True: It processes each image independently (static mode), which is suited for handling individual hand images.

DATA_DIR = './data'

data = []
labels = []
for class_dir in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, class_dir)):
        data_aux = []

        x_ = []
        y_ = []


        img = cv2.imread(os.path.join(DATA_DIR, class_dir, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(class_dir)

# f = open('data.pickle', 'wb')
# pickle.dump({'data': data, 'labels': labels}, f)
# f.close()