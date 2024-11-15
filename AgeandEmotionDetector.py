import cv2
import dlib
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from mtcnn import MTCNN

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
        self.face_detector = MTCNN()
    
    def load_emotion_model(self):
        model_path = os.path.join("Model", "emotion_model.h5")
        return load_model(model_path)
    
    def preprocess_image(self, image):
        # Detect and align face
        faces = self.face_detector.detect_faces(image)
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]['box']
        face = image[y:y+h, x:x+w]
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        roi = cv2.resize(gray, (48, 48))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(np.expand_dims(roi, axis=-1), axis=0)
        return roi
    
    def predict_emotion(self, face_img):
        """
        Predict the emotion from a given face image.
        """
        # Preprocess the input image
        roi = self.preprocess_image(face_img)
        
        if roi is None:
            return "No face detected"
        
        # Get emotion prediction probabilities
        preds = self.emotion_model.predict(roi)[0]
        
        # Map prediction to the emotion label
        return self.emotion_labels[np.argmax(preds)]
    
    
    
    