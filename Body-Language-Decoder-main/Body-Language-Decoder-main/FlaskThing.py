from sklearn.metrics import accuracy_score # Accuracy metrics
import pickle
import csv
import os
import numpy as np
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.http import JsonResponse
import json

@csrf_exempt
@require_POST

def post(request):
    data = json.loads(request.body.decode('utf-8'))
    pose_data = data.get('pose')
    face_data = data.get('face')

    print("Pose Data:")
    print(json.dumps(pose_data, indent=4))

    print("Face Data:")
    print(json.dumps(face_data, indent=4))

    return JsonResponse({'status': 'Success'})

def decompose(pose_data, face_data):
    print("Pose Data:")
    print(json.dumps(pose_data, indent=4))
    print("Face Data:")
    print(json.dumps(face_data, indent=4))


def predictDrowsy(model, pose, face):

    try:
        # Extract Pose landmarks
        landmarks = []
        for landmark in pose:
            if landmark is None:
                landmarks.append([0,0,0,0])
            else:
                landmarks.append([landmark['x'], landmark['y'], landmark['z'], 0])

        pose_row = list(np.array(landmarks).flatten())

        # Extract Face landmarks
        face_row = list(
            np.array([[landmark['x'], landmark['y'], landmark['z'], 0] for landmark in face]).flatten())
        print(face_row)
        # Concate rows
        row = face_row + pose_row

        print(row)
        # Make Detections
        X = pd.DataFrame([row])
        body_language_class = model.predict(X)[0]
        body_language_prob = model.predict_proba(X)[0]
        return body_language_prob, body_language_class
    except:
        return None, None


if __name__ == "__main__":
    with open('face.json', 'rb') as f:
        face_data = json.load(f)
    with open('pose.json', 'rb') as f:
        pose_data = json.load(f)
    with open('body_language.pkl', 'rb') as f:
        model = pickle.load(f)

    print(predictDrowsy(model, pose_data, face_data))