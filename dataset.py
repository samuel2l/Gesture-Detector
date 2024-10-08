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

DATA_DIR = '../data'

data = []
labels = []
for class_dir in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, class_dir)):
        normalised_cors = []

        x_cors = []
        y_cors = []


        img = cv2.imread(os.path.join(DATA_DIR, class_dir, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # mp_drawing.draw_landmarks(img_rgb,hand_landmarks,mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())
                
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_cors.append(x)
                    y_cors.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    normalised_cors.append(x - min(x_cors))
                    normalised_cors.append(y - min(y_cors))

            #the dataset then will comprise coordinates of the landmarks of the hand to predict the gesture
            data.append(normalised_cors)

            labels.append(class_dir)
        # plt.figure()

        # plt.imshow(img_rgb)
        # plt.show()

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()