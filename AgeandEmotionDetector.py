import cv2
import dlib
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

class AgeDetector:
    def __init__(self):
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
                         '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.model_mean = (78.4263377603, 87.7689143744, 114.895847746)
        self.age_net = self.load_age_detection_model()
        self.face_detector = dlib.get_frontal_face_detector()
    
    def load_age_detection_model(self):
        # Load the age detection model
        age_weights = os.path.join("Model", "age_deploy.prototxt")
        age_config = os.path.join("Model", "age_net.caffemodel")
        return cv2.dnn.readNet(age_config, age_weights)
    
    def detect_faces(self, frame):
        #Detect faces in the frame using dlib
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_detector(gray)
    
    def predict_age(self, face_img):
        #Predict the age of the detected face
        blob = cv2.dnn.blobFromImage(
            face_img, 1.0, (227, 227), self.model_mean, swapRB=False
        )
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        return self.age_list[age_preds[0].argmax()]
    
    
class EmotionDetector:
    
    def __init__(self):
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.emotion_model = self.load_emotion_model()
    
    def load_emotion_model(self):
        model_path = os.path.join("Model", "emotion_model.h5")
        return load_model(model_path)
    
    def predict_emotion(self, face_img):
        # Preprocess the image for emotion detection
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(gray, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        # Get emotion prediction
        preds = self.emotion_model.predict(roi)[0]
        return self.emotion_labels[np.argmax(preds)]
    
    
    
    