import streamlit as st
from PIL import Image
import cv2
from UIHelper import MainApp

def main():
    st.title("Age Detection")
    
    #Option for user to select input type
    input_type = st.radio(
        "Select Input Type",
        ("Live Feed", "Image")
    )
    video_source = 0
    if input_type == "Live Feed":
        video_source = st.selectbox("Select Video Source", [0, 1, 2], index= 0)
        
    app = MainApp(video_source=video_source)
    
    if input_type == "Image":
        upload_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
        
        if upload_file is not None:
            image = Image.open(upload_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            st.write("Detecting......")
            result_img = app.process_image(image)
            
            st.image(result_img, caption = "Processed Image", use_column_width=True )
            
    elif input_type == "Live Feed":
        stframe = st.empty()
        
        start_button = st.button("Start Camera")
        
        if start_button:
            while True:
                frame = app.process_live_feed()
                if frame is None:
                    st.write("Faild to grab frame")
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
                stframe.image(frame_rgb, channels="RGB")
            
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
    app.release_camera()
    

if __name__ == "__main__":
    main()