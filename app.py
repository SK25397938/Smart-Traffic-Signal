import streamlit as st
import tempfile
import os
import cv2
from ultralytics import YOLO
import numpy as np
import time
from PIL import Image

# ==============================
# --- CONFIGURATION & SETUP ---
# ==============================

# Initialize session state
if 'uploader_key' not in st.session_state:
    st.session_state['uploader_key'] = 0

if 'uploaded_image' not in st.session_state:
    st.session_state['uploaded_image'] = None

def reset_process():
    """Resets uploader and clears previous results."""
    st.session_state['uploader_key'] += 1
    st.session_state['uploaded_image'] = None # Clear image data
    st.rerun()  # ✅ Updated for new Streamlit versions

# Streamlit page setup
st.set_page_config(page_title="🚦 Smart Traffic Signal", page_icon="🚦", layout="centered")

# --- Custom CSS for the traffic light ---
st.markdown("""
<style>
.traffic-light {
    background-color: #2c3e50;
    border: 5px solid #34495e;
    border-radius: 20px;
    padding: 15px 10px;
    width: 100px;
    margin: 20px auto;
    box-shadow: 0 10px 20px rgba(0,0,0,0.5);
}
.light {
    width: 70px; height: 70px;
    border-radius: 50%;
    margin: 15px auto;
    border: 3px solid #1c2833;
    transition: all 0.3s ease-in-out;
}
.red { background-color: #ff0000; box-shadow: 0 0 30px 10px #ff0000; }
.yellow { background-color: #ffcc00; box-shadow: 0 0 30px 10px #ffcc00; }
.green { background-color: #00ff00; box-shadow: 0 0 30px 10px #00ff00; }
.off { background-color: #444; box-shadow: inset 0 0 15px rgba(0,0,0,0.8); }
</style>
""", unsafe_allow_html=True)

# --- Load YOLO model (cached) ---
@st.cache_resource
def load_model():
    try:
        # NOTE: Ensure 'yolov8n.pt' is in the same directory or accessible
        return YOLO("yolov8n.pt") 
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

model = load_model()
PERSON_CLASS_ID = 0

# ==============================
# --- CORE FUNCTIONS ---
# ==============================

def process_video(video_path):
    """Process the video and return max pedestrian count."""
    if model is None:
        return 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open video.")
        return 0

    max_count = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)
    status = st.empty()

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        # Process every 5th frame for performance
        if frame_idx % 5 != 0:
            progress.progress(frame_idx / frame_count)
            continue

        results = model(frame, verbose=False)
        for result in results:
            if result.boxes is not None:
                # YOLO output uses a tensor/numpy array for class IDs
                count = np.sum(result.boxes.cls.cpu().numpy() == PERSON_CLASS_ID)
                max_count = max(max_count, int(count))

        progress.progress(frame_idx / frame_count)
        status.text(f"Processing frame {frame_idx}/{frame_count} | Max detected: {max_count}")

    cap.release()
    progress.empty()
    status.empty()
    return max_count


def process_image_in_memory(pil_image):
    """Process an image (from PIL object) and return pedestrian count."""
    if model is None:
        return 0
    
    # Convert PIL Image to an OpenCV (BGR) format NumPy array
    img_np = np.array(pil_image.convert('RGB')) # PIL is RGB
    img = img_np[:, :, ::-1] # Convert RGB to BGR (OpenCV format)
    
    # Process the in-memory array
    results = model(img, verbose=False)
    count = 0
    for result in results:
        if result.boxes is not None:
            count = np.sum(result.boxes.cls.cpu().numpy() == PERSON_CLASS_ID)
    return int(count)


def calculate_timing(max_count, road_width, walking_speed):
    """
    Calculates pedestrian signal timing based on detected count and geometry.
    """
    # Formula: red_time = person * (road_width / avg_walking_speed)
    red_time = max_count * (road_width / walking_speed)
    yellow_time = 3 # Fixed 3 seconds for clearance
    
    total_time = red_time + yellow_time
    
    # Cap total time at 90 seconds (a common municipal standard)
    total_time = min(total_time, 90) 
    
    # Ensure red_time doesn't go below zero and respects the total time cap
    red_time = max(0, min(red_time, total_time - yellow_time))
    
    return {
        "max_count": max_count,
        "red_time": int(red_time),
        "yellow_time": yellow_time,
        "total_time": int(total_time)
    }


def display_traffic_light(state):
    red = "red" if state == "red" else "off"
    yellow = "yellow" if state == "yellow" else "off"
    green = "green" if state == "green" else "off"
    st.markdown(f"""
    <div class="traffic-light">
        <div class="light {red}"></div>
        <div class="light {yellow}"></div>
        <div class="light {green}"></div>
    </div>
    """, unsafe_allow_html=True)


def run_signal_cycle(timing):
    """
    Simulates the Signal Cycle: YELLOW (Warning) -> RED -> YELLOW (Clearance) -> GREEN
    """
    red_time = timing["red_time"]
    yellow_time = timing["yellow_time"]

    placeholder = st.empty()
    with st.status("🚦 Running Signal Cycle...", expanded=True) as status:
        
        # 1. Warning Yellow (Flowing traffic prepares to stop)
        status.update(label="🟡 Warning Phase (Prepare to Stop)", state="running")
        for i in range(yellow_time, 0, -1):
            with placeholder.container():
                st.metric("Time Remaining", f"{i}s")
                display_traffic_light("yellow")
            time.sleep(1)

        # 2. Red (Pedestrian Crossing)
        status.update(label="🔴 Red Phase (Pedestrian Crossing)", state="running")
        for i in range(red_time, 0, -1):
            with placeholder.container():
                st.metric("Time Remaining", f"{i}s")
                display_traffic_light("red")
            time.sleep(1)

        # 3. Clearance Yellow (Pedestrians clear, oncoming traffic prepares to flow)
        status.update(label="🟡 Clearance Phase (Prepare to Flow)", state="running")
        for i in range(yellow_time, 0, -1):
            with placeholder.container():
                st.metric("Time Remaining", f"{i}s")
                display_traffic_light("yellow")
            time.sleep(1)

        # 4. Back to Green
        status.update(label="🟢 Green - Normal Flow", state="complete", expanded=False)
        with placeholder.container():
            st.metric("Signal", "GREEN")
            display_traffic_light("green")


# ==============================
# --- STREAMLIT APP BODY ---
# ==============================

st.title("🚦Smart Traffic Signal")
st.markdown("Detect pedestrians using **YOLOv8** and compute smart signal timing based on road width and walking speed.")

st.button("🔄 Reset", on_click=reset_process, type="secondary")

# Input fields
col_input = st.columns(2)
with col_input[0]:
    road_width = st.number_input("Enter Road Width (meters)", min_value=1.0, value=10.0, step=0.5)
with col_input[1]:
    avg_walking_speed = st.number_input("Enter Avg Walking Speed (m/s)", min_value=0.5, value=1.4, step=0.1)

st.markdown("---")

# Upload area (photo or video)
media_type = st.radio("Choose Input Type", ["Video", "Image"])

# --- File Uploader ---
if media_type == "Video":
    uploaded = st.file_uploader("Upload road video", type=["mp4", "mov", "avi", "mkv"], key=st.session_state['uploader_key'])
else:
    # Use the same uploader, but handle image storage separately
    uploaded = st.file_uploader("Upload road image", type=["jpg", "jpeg", "png"], key=st.session_state['uploader_key'])
    # Handle image display and storage for in-memory processing
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        # Store the PIL image object in session state for later processing
        st.session_state['uploaded_image'] = image
    else:
        st.session_state['uploaded_image'] = None

# --- Analysis Button ---
if uploaded:
    if media_type == "Video" and uploaded:
        st.video(uploaded) # Display video after upload
        
    if st.button("▶️ Start AI Analysis"):
        count = 0
        path = None
        
        try:
            st.spinner("Processing...")
            
            if media_type == "Video":
                # --- Video: Must save to disk for cv2.VideoCapture ---
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
                    tmp.write(uploaded.read())
                    # Ensure file is closed before accessing (implicit with 'with' block)
                    path = tmp.name
                
                count = process_video(path)
                
            elif media_type == "Image" and st.session_state['uploaded_image']:
                # --- Image: Process in memory (no temp file needed) ---
                count = process_image_in_memory(st.session_state['uploaded_image'])
            
            # --- Calculation & Results ---
            timing = calculate_timing(count, road_width, avg_walking_speed)

            st.success("✅ Analysis Complete")
            st.metric("Max Pedestrians Detected", timing["max_count"])
            st.metric("Red Duration", f"{timing['red_time']} sec")
            st.metric("Total Cycle Duration", f"{timing['total_time']} sec (capped at 90s)")
            
            st.markdown("---")
            st.subheader("Signal Simulation")
            if timing["max_count"] > 0:
                run_signal_cycle(timing)
            else:
                st.info("No pedestrians detected — signal remains GREEN.")
                display_traffic_light("green")

        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            # Clean up the video temporary file if it was created
            if path and os.path.exists(path):
                os.unlink(path)

else:
    st.info("⬆️ Upload a video or photo to begin analysis.")