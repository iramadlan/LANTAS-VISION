import streamlit as st
import cv2
from ultralytics import YOLO, solutions
import gdown
import os
from tempfile import NamedTemporaryFile

# Function to download file from Google Drive
def download_from_google_drive(drive_url, output):
    file_id = drive_url.split('/d/')[1].split('/')[0]
    download_url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(download_url, output, quiet=False)

# Streamlit app
st.title("LANTAS-VISION Object Counting")

# Download the custom model
model_path = 'LANTAS-VISION.pt'
drive_link = 'https://drive.google.com/file/d/1j3FV8sq7BqGPU6Z-NInTVCRZif98HT-j/view?usp=sharing'

# Download model if it doesn't exist
if not os.path.exists(model_path):
    with st.spinner('Downloading model...'):
        download_from_google_drive(drive_link, model_path)
    st.success('Model downloaded!')

# Upload video
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None:
    tfile = NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    assert cap.isOpened(), "Error reading video file"
    
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Calculate proportional line points based on the frame dimensions
    line_y = int(h * 0.5)  # Line is placed at 50% of the frame height
    line_x_start = int(w * 0.05)  # Line starts at 5% of the frame width
    line_x_end = int(w * 0.95)  # Line ends at 95% of the frame width
    line_points = [(line_x_start, line_y), (line_x_end, line_y)]

    # Define classes
    classes_to_count = [0, 1, 2, 3, 4]  # Bus, Car, Motorbike, Person, Truck

    # Video writer setup
    output_path = "hasil_object_counting.mp4"
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Initialize Object Counter
    model = YOLO(model_path)
    counter = solutions.ObjectCounter(
        view_img=True,
        reg_pts=line_points,
        classes_names=model.names,
        draw_tracks=True,
        line_thickness=2,
    )

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            st.write("Video frame is empty or video processing has been successfully completed.")
            break
        tracks = model.track(im0, persist=True, show=False, classes=classes_to_count)

        # Start counting and update the frame with results
        im0 = counter.start_counting(im0, tracks)

        # Draw the line on the frame for visualization
        cv2.line(im0, line_points[0], line_points[1], (255, 0, 255), 2)

        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    # Display the output video
    st.video(output_path)

    st.success("Object counting completed and video saved!")
