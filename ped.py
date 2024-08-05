import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load the model
config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "frozen_inference_graph.pb"
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# Load class labels
classLabels = []
file_name = "labels.txt"
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

# Model settings
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

def detect_objects(image):
    img = np.array(image)
    ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)
    if len(ClassIndex) > 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if 0 < ClassInd <= len(classLabels):  # Ensure ClassInd is within valid range
                cv2.rectangle(img, boxes, (255, 0, 0), 2)
                cv2.putText(img, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), 
                            cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            else:
                print(f"Warning: ClassInd {ClassInd} is out of range")
    return img

st.title("Object Detection App")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Detecting objects...")
    detected_image = detect_objects(image)
    st.image(detected_image, caption='Detected Image.', use_column_width=True)

# Video detection
st.write("## Video Detection")

# Choose between webcam or video file
option = st.selectbox("Choose input source", ("Webcam", "Video File"))

if option == "Webcam":
    cap = cv2.VideoCapture(0)
else:
    video_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov", "mpeg4"], key="video_upload")
    if video_file is not None:
        video_path = video_file.name
        with open(video_path, 'wb') as out:
            out.write(video_file.read())
        cap = cv2.VideoCapture(video_path)
    else:
        cap = None

# Initialize session state for stop button
if 'stop' not in st.session_state:
    st.session_state.stop = False

if cap is not None:
    stop_button = st.button("Stop")
    if stop_button:
        st.session_state.stop = True

    stframe = st.empty()
    while cap.isOpened() and not st.session_state.stop:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_objects(frame)
        stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()

    # Reset the stop state
    st.session_state.stop = False
