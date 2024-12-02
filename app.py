#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import itertools
from collections import Counter, deque
import cv2 as cv
import numpy as np
import mediapipe as mp
from src.NN_History import PointHistoryClassifier
from src.NN_Classifier import KeyPointClassifier
from src.Cursor import Cursor

class GestureControlledPC:
    def __init__(self):
        self.cap = cv.VideoCapture(0)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.keypoint_classifier = KeyPointClassifier(model_path='./models/keypoint_classifier.tflite')
        self.point_history_classifier = PointHistoryClassifier(model_path='./models/point_history_classifier.tflite')

        self.keypoint_classifier_labels = self.load_labels('./data/NN/keypoint_classifier_label.csv')
        self.point_history_classifier_labels = self.load_labels('./data/NN/point_history_classifier_label.csv')

        self.history_length = 16    
        self.point_history = deque(maxlen=self.history_length)
        self.finger_gesture_history = deque(maxlen=self.history_length)
        self.cursor = Cursor(cap_height=540, cap_width=960)

    def load_labels(self, file_path):
        with open(file_path, encoding='utf-8-sig') as f:
            return [row[0] for row in csv.reader(f)]

    def process_hand_landmarks(self, image):
        results = self.hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = self.calc_bounding_rect(image, hand_landmarks)
                landmark_list = self.calc_landmark_list(image, hand_landmarks)
                pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
                pre_processed_point_history_list = self.pre_process_point_history(image, self.point_history)

                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                self.point_history.append(landmark_list[8] if hand_sign_id == 2 else [0, 0])

                finger_gesture_id = 0
                if len(pre_processed_point_history_list) == (len(self.point_history) * 2):
                    finger_gesture_id = self.point_history_classifier(pre_processed_point_history_list)
                self.finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(self.finger_gesture_history).most_common(1)[0][0]
                self.cursor.select(gid=hand_sign_id, mid=most_common_fg_id, landmarks=landmark_list)

                cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
                image = self.draw_landmarks(image, landmark_list)
                image = self.draw_info_text(image, brect, handedness, self.keypoint_classifier_labels[hand_sign_id], self.point_history_classifier_labels[most_common_fg_id])
        else:
            self.point_history.append([0, 0])
        return image

    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_points = [(min(int(landmark.x * image_width), image_width - 1), min(int(landmark.y * image_height), image_height - 1)) for landmark in landmarks.landmark]
        x, y, w, h = cv.boundingRect(np.array(landmark_points))
        return [x, y, x + w, y + h]

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        return [[min(int(landmark.x * image_width), image_width - 1), min(int(landmark.y * image_height), image_height - 1)] for landmark in landmarks.landmark]

    def pre_process_landmark(self, landmark_list):
        base_x, base_y = landmark_list[0]
        temp_landmark_list = [(x - base_x, y - base_y) for x, y in landmark_list]
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        max_value = max(map(abs, temp_landmark_list))
        return [x / max_value for x in temp_landmark_list]

    def pre_process_point_history(self, image, point_history):
        image_width, image_height = image.shape[1], image.shape[0]
        base_x, base_y = point_history[0]
        temp_point_history = [((x - base_x) / image_width, (y - base_y) / image_height) for x, y in point_history]
        return list(itertools.chain.from_iterable(temp_point_history))

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
        while True:
            if cv.waitKey(10) == 27: break  # ESC

            ret, image = self.cap.read()
            if not ret: break
            image = cv.flip(image, 1)
            debug_image = image.copy()

            debug_image = self.process_hand_landmarks(debug_image)
            cv.imshow('Hand Gesture Recognition', debug_image)

        self.cap.release()
        cv.destroyAllWindows()

if __name__ == '__main__':
    GestureControlledPC().run()
