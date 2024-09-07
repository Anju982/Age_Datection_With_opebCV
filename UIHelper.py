import cv2
import AgeDetector
import numpy as np


class MainApp:
    def __init__(self, video_source=0):
        self.age_detector = AgeDetector.AgeDetector()
        self.cap = cv2.VideoCapture(video_source)
        
        
    def display_result(self, frame, faces):
        
        if not faces:
            cv2.putText(frame, 'No face found', (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        else:
            for face in faces:
                x = face.left()
                y = face.top()
                x2 = face.right()
                y2 = face.bottom()
                
                cv2.rectangle(frame,(x,y),(x2,y2),(0,200,200),2)
                
                face_img = frame[y:y2, x:x2]
                age = self.age_detector.predict_age(face_img)
                
                cv2.putText(frame, f'Age: {age}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255), 2, cv2.LINE_AA)
                
        return frame
    
    def process_live_feed(self):
        ret, frame = self.cap.read()
        
        if not ret:
            return None
        
        faces = self.age_detector.detect_faces(frame)
        result_frame = self.display_result(frame, faces)
        
        return result_frame
    
    def process_image(self, image):
        image = np.array(image.convert('RGB'))
        faces = self.age_detector.detect_faces(image)
        result = self.display_result(image, faces)
        
        return result