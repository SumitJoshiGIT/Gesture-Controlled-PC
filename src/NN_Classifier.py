
import numpy as np
import tensorflow as tf


class KeyPointClassifier:
    def __init__(self, model_path, num_threads=1):
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.output_index = self.interpreter.get_output_details()[0]['index']

    def __call__(self, landmark_list):
        self.interpreter.set_tensor(self.input_index, np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()
        result = self.interpreter.get_tensor(self.output_index)
        return np.argmax(result)
