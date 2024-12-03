#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import itertools
from collections import Counter, deque
import cv2 as cv
import numpy as np
import mediapipe as mp
from  NN_History import PointHistoryClassifier
from  NN_Classifier import KeyPointClassifier
from  Cursor import Cursor

class GestureControlledPC:
    def __init__(self):
        self.cam = cv.VideoCapture(0)
        self.cam.set(cv.CAP_PROP_FRAME_WIDTH, 400)
        self.cam.set(cv.CAP_PROP_FRAME_HEIGHT, 400)
       
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.keypoint_classifier = KeyPointClassifier(model_path='./models/keypoint_classifier.tflite')
        self.point_history_classifier = PointHistoryClassifier(model_path='./models/point_history_classifier.tflite')

        self.keypoint_classifier_labels = self.load_labels('./data/NN/keypoint_classifier_label.csv')
        self.point_history_classifier_labels = self.load_labels('./data/NN/point_history_classifier_label.csv')

        self.history_length = 16    
        self.point_history = deque(maxlen=self.history_length)
        self.finger_gesture_history = deque(maxlen=self.history_length)
        self.cursor = Cursor(cap_height=400, cap_width=400)

    def load_labels(self, file_path):
        with open(file_path, encoding='utf-8-sig') as f:
            return [row[0] for row in csv.reader(f)]

    def process_hand_landmarks(self, image):
        results = self.hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmark_list = np.array([[int(landmark.x * 400), int(landmark.y * 400)] for landmark in hand_landmarks.landmark])

            base_x, base_y = landmark_list[0]
            temp_landmark_list = landmark_list - [base_x, base_y]
            max_value = np.max(np.abs(temp_landmark_list)) or 1
            pre_processed_landmark_list = (temp_landmark_list / max_value).flatten()

            if self.point_history:
             base_x, base_y = self.point_history[0]
             temp_point_history = (np.array(self.point_history) - [base_x, base_y]) / [400, 400]
             pre_processed_point_history_list = temp_point_history.flatten()
            else:pre_processed_point_history_list = []

            hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
            self.point_history.append(landmark_list[8] if hand_sign_id == 2 else [0, 0])
            finger_gesture_id = 0
            if len(pre_processed_point_history_list) == (self.history_length * 2):
             finger_gesture_id = self.point_history_classifier(pre_processed_point_history_list)
            self.finger_gesture_history.append(finger_gesture_id)
            most_common_fg_id = Counter(self.finger_gesture_history).most_common(1)[0][0]
            self.cursor.select(gid=hand_sign_id, mid=most_common_fg_id, landmarks=landmark_list)
        else:
            self.point_history.append([0, 0])
        return image

    # def calc_bounding_rect(self, image, landmarks):
    #     image heig = image.shape[1], image.shape[0]
    #     landmark_points = [(min(int(landmark.x * 400), 400 - 1), min(int(landmark.y * 400), 400 - 1)) for landmark in landmarks.landmark]
    #     x, y, w, h = cv.boundingRect(np.array(landmark_points))
    #     return [x, y, x + w, y + h]





    def draw_landmarks(self, image, landmark_point):
        if len(landmark_point) > 0:
            connections = [(2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)]
            for start, end in connections:
                cv.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]), (0, 0, 0), 6)
                cv.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]), (255, 255, 255), 2)
            for index, landmark in enumerate(landmark_point):
                radius = 8 if index in [4, 8, 12, 16, 20] else 5
                cv.circle(image, (landmark[0], landmark[1]), radius, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), radius, (0, 0, 0), 1)
        return image

    def draw_info_text(self, image, brect, handedness, hand_sign_text, finger_gesture_text):
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
        info_text = handedness.classification[0].label
        if hand_sign_text: info_text += f': {hand_sign_text}'
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        if finger_gesture_text:
            cv.putText(image, f" {finger_gesture_text}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
            cv.putText(image, f" {finger_gesture_text}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
        return image

    def run(self):
        print("FPS :")
        while True:
            ret, image = self.cam.read()
            if not ret: break
            debug_image = cv.flip(image, 1)
            debug_image = self.process_hand_landmarks(debug_image)
            #cv.imshow('Hand Gesture Recognition', debug_image)

        self.cam.release()
        cv.destroyAllWindows()

if __name__ == '__main__':
    GestureControlledPC().run()
