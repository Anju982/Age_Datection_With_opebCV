import cv2
import AgeDetector

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
    
    def run(self):
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            faces = self.age_detector.detect_faces(frame)
            frame = self.display_result(frame, faces)
            
            cv2.imshow("Age Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    app = MainApp(video_source=0) # Need to change Video source 
    app.run()

    