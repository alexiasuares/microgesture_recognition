import cv2
import mediapipe as mp
import pandas as pd 
import numpy as np
import joblib
import time
import json
#import os

model = joblib.load('models/attention_detection_model.pkl')
with open('models/feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_features(face_landmarks):
    coords = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]

    features = {}
    features["eye_distance"] = np.linalg.norm(np.array(coords[33][:2]) - np.array(coords[263][:2]))
    features["mouth_open"] = np.linalg.norm(np.array(coords[13][:2]) - np.array(coords[14][:2]))
    