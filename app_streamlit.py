import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from ultralytics import YOLO
import cv2
import json
import math
import requests

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

# Function to calculate distance between two coordinates
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

# Get the closest STUN server based on user's geolocation
def get_closest_stun_server():
    GEO_LOC_URL = "https://raw.githubusercontent.com/pradt2/always-online-stun/master/geoip_cache.txt"
    IPV4_URL = "https://raw.githubusercontent.com/pradt2/always-online-stun/master/valid_ipv4s.txt"
    GEO_USER_URL = "https://geolocation-db.com/json/"

    geoLocs = requests.get(GEO_LOC_URL).json()
    user_location = requests.get(GEO_USER_URL).json()
    latitude = user_location['latitude']
    longitude = user_location['longitude']

    stun_servers = requests.get(IPV4_URL).text.strip().split('\n')
    closest_server = None
    min_distance = float('inf')

    for server in stun_servers:
        server_ip = server.split(':')[0]
        if server_ip in geoLocs:
            stun_lat, stun_lon = geoLocs[server_ip]
            distance = calculate_distance(latitude, longitude, stun_lat, stun_lon)
            if distance < min_distance:
                min_distance = distance
                closest_server = server

    return closest_server

# Get the closest STUN server
closest_stun_server = get_closest_stun_server()



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
        "iceServers": [{"urls": f"stun:{closest_stun_server}"}]},
    media_stream_constraints={"video": video_stream_constraints, "audio": False}
)
