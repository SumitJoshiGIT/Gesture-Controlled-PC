
import pyautogui

class Actions:
    def __init__(self):
        pass
    
    def record_gesture(self):
        pass

    def predict_gesture(self):
        pass     

    def  move_cursor(self,x,y):
        pass

    def  press_key(self,key):
        pass

if __name__ == "__main__":
    action = Actions()
    action.press_key('a')
    action.move_cursor()
    action.record_gesture()
    action.predict_gesture()