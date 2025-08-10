import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import joblib
import json
import time


model = joblib.load('models/attention_detection_model.pkl')

with open('models/feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_features(frame, results_face, results_pose, results_hands):
    features = {
        'no_of_face': 0,
        'face_x': 0.0,
        'face_y': 0.0,
        'face_w': 0.0,
        'face_h': 0.0,
        'face_con': 0.0,
        'no_of_hand': 0,
        'pose_x': 0.0,
        'pose_y': 0.0
    }

    # Face
    if results_face and results_face.multi_face_landmarks:
        features['no_of_face'] = len(results_face.multi_face_landmarks)
        lm = results_face.multi_face_landmarks[0].landmark
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        features['face_x'] = min_x
        features['face_y'] = min_y
        features['face_w'] = max_x - min_x
        features['face_h'] = max_y - min_y
        features['face_con'] = 1.0 

    # Hands
    if results_hands and results_hands.multi_hand_landmarks:
        features['no_of_hand'] = len(results_hands.multi_hand_landmarks)

    # Pose
    if results_pose and results_pose.pose_landmarks:
        nose = results_pose.pose_landmarks.landmark[0]
        features['pose_x'] = nose.x
        features['pose_y'] = nose.y

    return features

cap = cv2.VideoCapture(0)

prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processamento com MediaPipe
    results_face = face_mesh.process(frame_rgb)
    results_pose = pose.process(frame_rgb)
    results_hands = hands.process(frame_rgb)

    # Extração de features
    feats = extract_features(frame, results_face, results_pose, results_hands)
    feats_df = pd.DataFrame([feats], columns=feature_columns)

    # Previsão do modelo
    prediction = model.predict(feats_df)[0]
    label_map = {0: 'Atento', 1: 'Cansado'}
    label = label_map.get(prediction, 'Desconhecido')

    # Mostrar resultado na tela
    cv2.putText(frame, f"Estado: {label}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Desenhar landmarks
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face.FACEMESH_CONTOURS)
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Detecção em Tempo Real", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Tecla ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
