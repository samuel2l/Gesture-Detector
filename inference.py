import pickle
import cv2
import easyocr
import mediapipe as mp
import numpy as np
import pyttsx3

model_dict = pickle.load(open('/Users/samuel/gesture detector/model.p', 'rb'))
model = model_dict['model']
engine = pyttsx3.init()
labels_dict = {0: 'Live Long man', 1: 'Thank you bro', 2: 'You suck', 3: 'Oi start'}

reader = easyocr.Reader(['en'], gpu=False)
cap = cv2.VideoCapture(0)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

max_length = 84  # Set according to your training data's maximum length

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape


    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS, 
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

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

            if len(data_aux) < max_length:
                data_aux.extend([0] * (max_length - len(data_aux)))  # Pad with zeros
            else:
                data_aux = data_aux[:max_length]

            data_aux = np.array(data_aux).reshape(1, -1)  # Reshape for model input

            prediction = model.predict(data_aux)


            gesture_pred = labels_dict[int(prediction[0])]
#add nice little feature where if you make a specific gesture then detect text in ssome image and read it out
            if gesture_pred == 'Oi start':  # You can change to any gesture
            # Now activate text detection using OCR
                print("Gesture detected! Activating text detection...")

                # Detect text in the current frame
                image_path = '/Users/samuel/realtime_obj_detection/obj detection/iimg.jpg'

                img = cv2.imread(image_path)
                text_ = reader.readtext(img)

                if text_:
                    print(f"Text detected: {text_}")

                    for t in text_:
                        bbox, detected_text, score = t

                        # Draw a bounding box around the detected text
                        cv2.rectangle(frame, bbox[0], bbox[2], (0, 255, 0), 2)
                        cv2.putText(frame, detected_text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

                        # Use TTS to read out the detected text
                        engine.say(detected_text)
                        engine.runAndWait()



            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, gesture_pred, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
