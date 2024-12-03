import pyautogui 
import numpy as np
import time
class Cursor():
    def __init__(self,cap_width,cap_height):
        self.screen_width,self.screen_height=pyautogui.size()
        self.cap_width = cap_width
        self.cap_height = cap_height
        self.time = 0
    def select(self,gid,mid,landmarks):
        #['Open', 'Close', 'Pointer', 'OK']
        #['Stop', 'Clockwise', 'Counter Clockwise', 'Move']
       
        if(mid==3):self.move(landmarks)
        #elif(mid==1):pyautogui.scroll(50)  
        #elif(mid==2):pyautogui.scroll(-50)
        elif(gid==1):self.click()
        elif(gid==3):self.scroll(landmarks)
         #Scroll Up By 10 Units # Scroll Down By 10 Units 
         #Move Cursor To The Right # Move Cursor To The Left #
        #elif(gid==0):pyautogui.press(button='left')
    def click(self):   
        if(time.time()-self.time>1.5):  # To avoid multiple clicks
           
           pyautogui.click(button='left')
        self.time=time.time()     
    def scroll(self,landmarks):
            thumb_x, thumb_y = landmarks[4]  # Thumb tip
            index_x, index_y = landmarks[8]  # Index finger tip
            distance = np.linalg.norm(np.array([thumb_x, thumb_y]) - np.array([index_x, index_y]))  
            if distance < 50:pyautogui.scroll(50)
            else:pyautogui.scroll(-50)    

    def move(self,landmark_list):
        x, y = landmark_list[8]  # Using the tip of the index finger
        x = np.clip(x * self.screen_width / self.cap_width, 0, self.screen_width)
        y = np.clip(y * self.screen_height / self.cap_height, 0, self.screen_height)
        
        current_x, current_y = pyautogui.position()
        smooth_x = current_x + (x - current_x) / 5
        smooth_y = current_y + (y - current_y) / 5
        
        pyautogui.moveTo(smooth_x, smooth_y)
