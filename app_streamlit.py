import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from ultralytics import YOLO
import cv2

model = YOLO("best_v8_25.pt")  # Path to the pre-trained YOLOv5 nano model

st.title("Real-Time Object Detection with YOLOv8")



# Define a function to draw bounding boxes with a specific color
def draw_bounding_boxes(img, results):
     # Initialize counters
    helmet_count = 0
    no_helmet_count = 0
    for result in results:
        for box in result.boxes:
            bbox = box.xyxy[0].cpu().numpy().astype(int)  # Get the bounding box coordinates
            class_id = int(box.cls[0])
            confidence = box.conf[0]

            # Define the color based on the class label (class 0 is helmet and class 1 is no-helmet)
            if class_id == 1:
                color = (0, 0, 255)  # Red for no helmet
                label = f"No Helmet: {confidence:.2f}"
                no_helmet_count += 1
            else:
                color = (0, 255, 0)  # Green for helmet
                label = f"Helmet: {confidence:.2f}"
                helmet_count += 1

            # Draw the bounding box and label on the image
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(img, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Display the counts on the frame
            cv2.putText(img, f"Helmet: {helmet_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"No Helmet: {no_helmet_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return img


# Callback function to process video frames
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Get model predictions
    results = model(img)

    # Draw bounding boxes on the image
    img_with_boxes = draw_bounding_boxes(img, results)

    return av.VideoFrame.from_ndarray(img_with_boxes, format="bgr24")

video_stream_constraints = {
            "width": {"ideal": 640},  # Lower resolution to reduce bandwidth
            "height": {"ideal": 480},
            "frameRate": {"ideal": 15, "max": 30}
            }

webrtc_streamer(
    key="example",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stunserver.org:3478"]},
                       {"urls": ["stun.xtratelecom.es:3478"]},
                       {"urls": ["stun.wifirst.net:3478"]},
                       ]},
    media_stream_constraints={"video": video_stream_constraints, "audio": False}
)
