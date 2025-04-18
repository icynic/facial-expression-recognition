import csv
import copy
import itertools
import os
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

def encode_label(label_name,category):
    for i in category:
        if i == label_name:
            return category.index(i)


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark_revised(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # 1. Translate relative to a stable point (e.g., nose tip - index 1)
    if not temp_landmark_list:
        return []
        
    base_x, base_y = temp_landmark_list[1][0], temp_landmark_list[1][1]

    for landmark_point in temp_landmark_list:
        landmark_point[0] -= base_x
        landmark_point[1] -= base_y

    # 2. Calculate scale factor (e.g., inter-ocular distance)
    # Use coordinates *after* translation for scale calculation as well
    p1 = temp_landmark_list[263] # Left eye outer corner
    p2 = temp_landmark_list[33]  # Right eye outer corner
    
    # Add small epsilon to prevent division by zero
    scale_factor = np.linalg.norm(np.array(p1) - np.array(p2)) + 1e-6 

    # 3. Normalize coordinates by the scale factor
    normalized_landmarks = []
    for landmark_point in temp_landmark_list:
        normalized_landmarks.append(landmark_point[0] / scale_factor)
        normalized_landmarks.append(landmark_point[1] / scale_factor)
        
    # 4. Flattening is already done by appending x, y sequentially

    return normalized_landmarks


def logging_csv(number, landmark_list):
    assert isinstance(number, int)
    # Empty data if it already exists
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *landmark_list])
    return


# root = "dataset/FER-2013/train"
# root = "dataset/CK+48"
# root = "dataset/affectnet"
root="dataset/jaffe/reorganized"

csv_path = 'model/keypoint_classifier/keypoint.csv'
# Clear existing data
with open(csv_path, 'w', newline="") as f:
    pass

IMAGE_FILES = []
# category = ['angry','happy','neutral','sad','surprise', 'disgust', 'fear']
# category = ['anger','contempt','disgust','fear','happy','neutral','sad','surprise']
category=[]
for item in Path(root).iterdir():
    if item.is_dir():
        category.append(item.name)


for path, subdirs, files in os.walk(root):
    for name in files:
        IMAGE_FILES.append(os.path.join(path, name))

use_brect = True

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        static_image_mode=True) 


for idx, file in enumerate(IMAGE_FILES):
    parent_dir = os.path.dirname(file)
    label_name = os.path.basename(parent_dir)
    label = encode_label(label_name,category)
    image = cv2.imread(file)
    
    # Check if image was loaded successfully
    if image is None:
        print(f"Warning: Could not read image file {file}. Skipping.")
        continue # Skip to the next file

    image = cv2.flip(image, 1)  # Mirror display
    debug_image = copy.deepcopy(image)

    # Detection implementation
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:

            # Landmark calculation
            landmark_list = calc_landmark_list(debug_image, face_landmarks)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark_revised(
                landmark_list)
            # Write to the dataset file
            logging_csv(label, pre_processed_landmark_list)