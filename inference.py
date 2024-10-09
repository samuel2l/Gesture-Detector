import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model from the pickle file
model_dict = pickle.load(open('/Users/samuel/gesture detector/model.p', 'rb'))
model = model_dict['model']

# Labels corresponding to predictions
labels_dict = {0: 'Live Long man', 1: 'Thank you bro', 2: 'You suck', 3: 'Oi start'}

# Capture video from the camera
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Assume max_length is the length of the largest sample used during training
max_length = 84  # Set according to your training data's maximum length

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    # Convert the frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS, 
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Collect normalized landmark data
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize the data by subtracting the minimum values
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # **Padding**: Pad the data to match the max_length used during training
            # This ensures the input to the model is always the same size
            if len(data_aux) < max_length:
                data_aux.extend([0] * (max_length - len(data_aux)))  # Pad with zeros
            else:
                data_aux = data_aux[:max_length]  # Truncate if longer than max_length

            # Convert to NumPy array
            data_aux = np.array(data_aux).reshape(1, -1)  # Reshape for model input

            # Predict the gesture
            prediction = model.predict(data_aux)

            # Get the label corresponding to the predicted class
            predicted_character = labels_dict[int(prediction[0])]

            # Get bounding box coordinates for visualization
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Draw the prediction on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Break the loop on keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
