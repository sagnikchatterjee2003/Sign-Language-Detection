import cv2
import pickle
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_tracking_confidence=0.3, max_num_hands=1)

labels_dict = {
    'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D'
}

while True:
    ret, frame = cap.read()

    data_aux = []

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)

        prediction = model.predict(np.asarray(data_aux).reshape(1, -1))

        predicted_character = labels_dict[prediction[0].lower()]

        cv2.putText(frame, predicted_character, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('`'):
        break

cap.release()
cv2.destroyAllWindows()
