import pyautogui 
import numpy as np
import time
class Cursor:
    def __init__(self, cap_width, cap_height):
        self.screen_width, self.screen_height = pyautogui.size()
        self.cap_width = cap_width
        self.cap_height = cap_height
        self.time = 0

    def select(self, gid, mid, landmarks):
        if mid == 3:
            self.move(landmarks)
        elif gid == 1:
            self.click()
        elif gid == 3:
            self.scroll(landmarks)

    def click(self):
        if time.time() - self.time > 1.5:
            pyautogui.click(button='left')
            self.time = time.time()

    def scroll(self, landmarks):
        thumb_x, thumb_y = landmarks[4]
        index_x, index_y = landmarks[8]
        distance = np.linalg.norm(np.array([thumb_x, thumb_y]) - np.array([index_x, index_y]))
        pyautogui.scroll(50 if distance < 50 else -50)

    def move(self, landmark_list):
        x, y = landmark_list[8]
        x = np.clip(x * self.screen_width / self.cap_width, 0, self.screen_width)
        y = np.clip(y * self.screen_height / self.cap_height, 0, self.screen_height)
        current_x, current_y = pyautogui.position()
        pyautogui.moveTo(current_x+(x-current_x),current_y+(y-current_y) )
        #     time.sleep(0.01)
