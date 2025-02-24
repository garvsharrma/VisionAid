import cv2
import torch
from ultralytics import YOLO
import pyttsx3
import speech_recognition as sr
from threading import Thread, Event
import queue
import time
import sys
from urllib.parse import urlparse
import socket

# Define threatening objects and their safe distances
THREAT_OBJECTS = {
    'person': 200,      # cm
    'car': 500,
    'motorcycle': 400,
    'bicycle': 300,
    'truck': 500,
    'pole': 150,
    'fire hydrant': 100,
    'stop sign': 150,
    'bench': 150,
    'chair': 150,
    'couch': 150,
}

KNOWN_WIDTH = 45  # cm (average human width)
FOCAL_LENGTH = 700 

def get_position(x_min, x_max, frame_width):
    center = (x_min + x_max) / 2
    third = frame_width / 3
    if center < third:
        return "on your left"
    elif center > 2 * third:
        return "on your right"
    return "in front of you"

def listen_for_commands(command_queue, stop_event):
    recognizer = sr.Recognizer()
    while not stop_event.is_set():
        with sr.Microphone() as source:
            try:
                print("Listening...")
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                command = recognizer.recognize_google(audio).lower()
                
                if "walking mode" in command:
                    command_queue.put("walking")
                elif "normal mode" in command:
                    command_queue.put("normal")
                elif any(trigger in command for trigger in ["what's around", "what is around", "describe"]):
                    command_queue.put("describe")
            except (sr.WaitTimeoutError, sr.UnknownValueError):
                continue
            except Exception as e:
                print(f"Error in voice recognition: {e}")
                continue

def test_connection(url):
    parsed = urlparse(url)
    try:
        sock = socket.create_connection((parsed.hostname, parsed.port), timeout=2)
        sock.close()
        return True
    except:
        return False

def connect_camera(url, max_retries=3):
    for attempt in range(max_retries):
        if test_connection(url):
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                print(f"Successfully connected to camera on attempt {attempt + 1}")
                # Improved camera settings for stability
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Increased buffer
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for stability
                
                # Warm up the connection
                for _ in range(10):
                    ret = cap.grab()
                    if not ret:
                        break
                return cap
        print(f"Connection attempt {attempt + 1} failed. Retrying in 2 seconds...")
        time.sleep(2)
    return None

# Initialize components
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 180)
model = YOLO('yolo11n.pt')

# Setup command handling
command_queue = queue.Queue()
stop_event = Event()
voice_thread = Thread(target=listen_for_commands, args=(command_queue, stop_event))
voice_thread.daemon = True
voice_thread.start()

# Initialize IP camera with retry mechanism
url = "http://192.168.157.43:8080/video"
print("Attempting to connect to IP camera...")
cap = connect_camera(url)

if cap is None:
    print("Error: Could not connect to IP camera after multiple attempts.")
    print("Please check:")
    print("1. IP camera app is running on phone")
    print("2. Phone and computer are on same network")
    print("3. IP address is correct")
    print("4. Port 8080 is accessible")
    sys.exit(1)

# Add mode tracking
current_mode = "normal"
last_announcement_time = time.time()
ANNOUNCEMENT_COOLDOWN = 2  # seconds
frame_skip = 4
frame_count = 0
prev_time = time.time()
frame_timeout = 5.0  # Increased timeout
consecutive_failures = 0
MAX_FAILURES = 3
last_reconnect_time = time.time()
RECONNECT_COOLDOWN = 5  # Minimum seconds between reconnection attempts
last_frame_time = time.time()

while True:
    try:
        ret, frame = cap.read()
        current_time = time.time()
        
        # More robust connection checking
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= MAX_FAILURES:
                if (current_time - last_reconnect_time) > RECONNECT_COOLDOWN:
                    print("Connection lost. Attempting to reconnect...")
                    cap.release()
                    cap = connect_camera(url)
                    last_reconnect_time = current_time
                    consecutive_failures = 0
                    if cap is None:
                        print("Reconnection failed. Exiting.")
                        break
            continue
        else:
            consecutive_failures = 0  # Reset counter on successful frame
            
        last_frame_time = current_time

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
            
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        frame = cv2.resize(frame, (640, 480))
        frame_width = frame.shape[1]
        
        results = model(frame, verbose=False, iou=0.8)[0]
        detected_speech = []
        threats = []

        for result in results.boxes.data:
            x_min, y_min, x_max, y_max, conf, class_id = result
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            obj_name = model.names[int(class_id)]
            
            if conf < 0.2:
                continue

            obj_width = x_max - x_min
            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / obj_width
            distance = round(distance)
            position = get_position(x_min, x_max, frame_width)

            # Check if object is a threat
            if obj_name in THREAT_OBJECTS and distance < THREAT_OBJECTS[obj_name]:
                threats.append((obj_name, distance, position))
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            cv2.putText(frame, f"{obj_name}: {distance}cm", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            detected_speech.append(f"{obj_name} {position} at {distance} centimeters")

        # Handle voice commands and modes
        try:
            if not command_queue.empty():
                command = command_queue.get_nowait()
                if command in ["walking", "normal"]:
                    current_mode = command
                    tts_engine.say(f"Switching to {command} mode")
                    tts_engine.runAndWait()
                elif command == "describe":
                    announcement = ", ".join(detected_speech)
                    tts_engine.say(announcement)
                    tts_engine.runAndWait()
        except queue.Empty:
            pass

        # Handle walking mode announcements
        if current_mode == "walking" and threats and (current_time - last_announcement_time) > ANNOUNCEMENT_COOLDOWN:
            threats.sort(key=lambda x: x[1])
            threat_announcements = [f"{obj} {pos} at {dist} centimeters" for obj, dist, pos in threats[:2]]
            if threat_announcements:
                announcement = "Warning: " + ", ".join(threat_announcements)
                tts_engine.say(announcement)
                tts_engine.runAndWait()
                last_announcement_time = current_time

        # Display mode and FPS
        cv2.putText(frame, f"Mode: {current_mode}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Obstacle Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error occurred: {e}")
        if (current_time - last_reconnect_time) > RECONNECT_COOLDOWN:
            print("Attempting to reconnect...")
            cap = connect_camera(url)
            last_reconnect_time = current_time
            if cap is None:
                print("Reconnection failed. Exiting.")
                break
        continue