#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import cv2 as cv
import numpy as np
import mediapipe as mp
from utils.cvfpscalc import CvFpsCalc
from src.NN_History import PointHistoryClassifier
from src.NN_Classifier import KeyPointClassifier
from src.Cursor import Cursor
def main():
    cap_device = 0
    cap_width = 960
    cap_height = 540
    use_static_image_mode = False
    min_detection_confidence = 0.7
    min_tracking_confidence = 0.5
    
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

 
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier(model_path='./models/keypoint_classifier.tflite')
    point_history_classifier = PointHistoryClassifier(model_path='./models/point_history_classifier.tflite')
    with open('./data/NN/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
    with open('./data/NN/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = [row[0] for row in csv.reader(f)]
    print(keypoint_classifier_labels)
    print( point_history_classifier_labels)
    # FPS Measurement 
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    history_length = 16    
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)
    cursor=Cursor(cap_height=cap_height, cap_width=cap_width)
    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(10)
        if key == 27: break  # ESC

        ret, image = cap.read()
        if not ret: break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                point_history.append(landmark_list[8] if hand_sign_id == 2 else [0, 0])

                finger_gesture_id = 0
                if len(pre_processed_point_history_list) == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common(1)[0][0]
                cursor.select(gid=hand_sign_id,mid=most_common_fg_id,landmarks=landmark_list) 
      
                #cv.rectangle(debug_image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
                #debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(debug_image, brect, handedness, keypoint_classifier_labels[hand_sign_id], point_history_classifier_labels[most_common_fg_id])
        else:
            point_history.append([0, 0])
        #print(point_history)
        #debug_image = draw_point_history(debug_image, point_history)
        #cv.putText(debug_image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        #cv.putText(debug_image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()



def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_points = [
        (min(int(landmark.x * image_width), image_width - 1),
         min(int(landmark.y * image_height), image_height - 1))
        for landmark in landmarks.landmark
    ]

    x, y, w, h = cv.boundingRect(np.array(landmark_points))

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    base_x, base_y = landmark_list[0]
    temp_landmark_list = [(x - base_x, y - base_y) for x, y in landmark_list]
    # Convert to a one-dimensional list and normalize
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(map(abs, temp_landmark_list))
    temp_landmark_list = [x / max_value for x in temp_landmark_list]
    return temp_landmark_list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    # Convert to relative coordinates and flatten the list
    base_x, base_y = point_history[0]
    temp_point_history = [((x - base_x) / image_width, (y - base_y) / image_height) for x, y in point_history]
    return list(itertools.chain.from_iterable(temp_point_history))

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Define connections for drawing lines
        connections = [
            (2, 3), (3, 4),  # Thumb
            (5, 6), (6, 7), (7, 8),  # Index finger
            (9, 10), (10, 11), (11, 12),  # Middle finger
            (13, 14), (14, 15), (15, 16),  # Ring finger
            (17, 18), (18, 19), (19, 20),  # Little finger
            (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)  # Palm
        ]

        for start, end in connections:
            cv.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]), (255, 255, 255), 2)

        for index, landmark in enumerate(landmark_point):
            radius = 8 if index in [4, 8, 12, 16, 20] else 5
            cv.circle(image, (landmark[0], landmark[1]), radius, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), radius, (0, 0, 0), 1)

    return image




def draw_info_text(image, brect, handedness, hand_sign_text,finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    info_text = handedness.classification[0].label
    if hand_sign_text:info_text += f': {hand_sign_text}'
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text:
        cv.putText(image, f"Finger Gesture: {finger_gesture_text}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, f"Finger Gesture: {finger_gesture_text}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),(152, 251, 152), 2)
    return image



if __name__ == '__main__':
    main()
