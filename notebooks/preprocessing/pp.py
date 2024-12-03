import cv2
import mediapipe as mp
import time
import csv
import os
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
hand_images_dir = "hand_images"
points_dir = "hand_points"
os.makedirs(hand_images_dir, exist_ok=True)
os.makedirs(points_dir, exist_ok=True)
last_time = time.time()
csv_filename = "hand_coordinates_with_gesture.csv"
frame_counter = 1  # Frame counter for image naming
header = ["Gesture", "Image Name", "T0", "T1", "T2", "T3", "Thumb", "I3", "I2", "I1", "Index", 
          "M3", "M2", "M1", "Middle", "R3", "R2", "R1", "Ring", "L3", "L2", "L1", "Little"]
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty frame.")
        continue
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        current_time = time.time()
        if current_time - last_time >= 0.5:
            last_time = current_time
            for hand_landmarks in results.multi_hand_landmarks:
                all_points_coordinates = []
                black_canvas = frame.copy()
                black_canvas[:, :] = 0 
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    all_points_coordinates.append((x, y))
                    cv2.circle(black_canvas, (x, y), 2, (0, 255, 0), -1)  # Reduced circle size
                hand_image_name = os.path.join(hand_images_dir, f"hand_{frame_counter}.png")
                point_image_name = os.path.join(points_dir, f"points_{frame_counter}.png")
                cv2.imwrite(hand_image_name, frame)
                cv2.imwrite(point_image_name, black_canvas)
                point_names = ["T0", "T1", "T2", "T3", "Thumb", "I3", "I2", "I1", "Index","M3", "M2", "M1", "Middle", "R3", "R2", "R1", "Ring", "L3", "L2", "L1", "Little"]
                row_data = ["gesture", hand_image_name] + [
                    f"{all_points_coordinates[i][0]},{all_points_coordinates[i][1]}" if i < len(all_points_coordinates) else ""
                    for i in range(len(point_names))]
                with open(csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row_data)
                frame_counter += 1
                print("Gesture, Image Name, and All Points' Coordinates:", row_data)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):break
cap.release()
cv2.destroyAllWindows()