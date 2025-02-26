import cv2
import torch
from ultralytics import YOLO
import pyttsx3
import speech_recognition as sr
import numpy as np
from threading import Thread, Event
import queue
import time
import pygame
import math
from scipy.spatial import distance as dist

# Enhanced threat definitions with more context
THREAT_OBJECTS = {
    'person': {'safe_distance': 200, 'priority': 5, 'motion_sensitive': True},
    'car': {'safe_distance': 500, 'priority': 8, 'motion_sensitive': True},
    'motorcycle': {'safe_distance': 400, 'priority': 7, 'motion_sensitive': True},
    'bicycle': {'safe_distance': 300, 'priority': 6, 'motion_sensitive': True},
    'truck': {'safe_distance': 500, 'priority': 8, 'motion_sensitive': True},
    'pole': {'safe_distance': 150, 'priority': 4, 'motion_sensitive': False},
    'fire hydrant': {'safe_distance': 100, 'priority': 3, 'motion_sensitive': False},
    'stop sign': {'safe_distance': 150, 'priority': 4, 'motion_sensitive': False},
    'bench': {'safe_distance': 150, 'priority': 3, 'motion_sensitive': False},
    'hole': {'safe_distance': 200, 'priority': 7, 'motion_sensitive': False},
    'stairs': {'safe_distance': 200, 'priority': 6, 'motion_sensitive': False},
}

# Add wall detection classes
SURFACE_OBJECTS = {
    'wall': {'min_area': 20000, 'danger_distance': 150},
    'floor': {'min_area': 30000, 'danger_distance': 100},
    'ceiling': {'min_area': 20000, 'danger_distance': 200},
    'door': {'min_area': 15000, 'danger_distance': 150},
}

# Update THREAT_OBJECTS with more structural hazards
THREAT_OBJECTS.update({
    'wall': {'safe_distance': 200, 'priority': 7, 'motion_sensitive': False},
    'door': {'safe_distance': 200, 'priority': 6, 'motion_sensitive': False},
    'stairs': {'safe_distance': 300, 'priority': 8, 'motion_sensitive': False},
    'elevator': {'safe_distance': 200, 'priority': 7, 'motion_sensitive': False},
})

# Add spatial audio feedback
def init_audio():
    pygame.mixer.init()
    pygame.mixer.set_num_channels(8)
    audio_path = "assets/audio"
    return {

        'warning': pygame.mixer.Sound(f"{audio_path}/warning.wav"),
        'proximity': pygame.mixer.Sound(f"{audio_path}/proximity.wav"),
        'direction': pygame.mixer.Sound(f"{audio_path}/direction.wav")
    }

def play_spatial_sound(sound, position, distance):
    # Convert position to stereo volume (left/right balance)
    left_vol = 1.0 - (position / frame_width)
    right_vol = position / frame_width
    # Distance affects overall volume
    volume = 1.0 - min(1.0, distance / 500)
    sound.set_volume(volume)
    # Set left/right channels
    pygame.mixer.Channel(0).set_volume(left_vol, right_vol)
    pygame.mixer.Channel(0).play(sound)

class MotionDetector:
    def __init__(self):
        self.previous_frame = None
        self.motion_threshold = 30

    def detect_motion(self, frame, bbox):
        if self.previous_frame is None:
            self.previous_frame = frame
            return False

        # Extract region of interest
        x1, y1, x2, y2 = bbox
        current_roi = frame[y1:y2, x1:x2]
        previous_roi = self.previous_frame[y1:y2, x1:x2]
        
        # Calculate motion
        if current_roi.size > 0 and previous_roi.size > 0:
            diff = cv2.absdiff(current_roi, previous_roi)
            motion = np.mean(diff)
            return motion > self.motion_threshold
        return False

def estimate_depth(width, height, area):
    # Improved depth estimation using multiple parameters
    depth_from_width = (KNOWN_WIDTH * FOCAL_LENGTH) / width
    depth_from_height = (KNOWN_WIDTH * FOCAL_LENGTH) / height
    depth_from_area = math.sqrt((KNOWN_WIDTH * FOCAL_LENGTH) / area)
    return (depth_from_width + depth_from_height + depth_from_area) / 3

def analyze_environment(frame):
    # Basic environment analysis
    brightness = np.mean(frame)
    contrast = np.std(frame)
    return {
        'low_light': brightness < 50,
        'high_contrast': contrast > 50,
        'motion_blur': detect_blur(frame)
    }

def detect_blur(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < 100

def get_position(x_min, x_max, frame_width):
    center = (x_min + x_max) / 2
    third = frame_width / 3
    if center < third:
        return "on your left"
    elif center > 2 * third:
        return "on your right"
    return "in front of you"

def analyze_path_clearance(frame_width, detected_objects):
    """Analyze the path ahead and find the maximum safe distance"""
    # Define path zones (left, center, right)
    zones = {
        'center': (frame_width/3, 2*frame_width/3),
        'left': (0, frame_width/3),
        'right': (2*frame_width/3, frame_width)
    }
    
    zone_clearances = {zone: float('inf') for zone in zones.keys()}
    
    for obj, distance, position, _ in detected_objects:
        x_center = position[0]  # Assuming position now returns (x_center, distance)
        # Check which zone the object is in
        for zone, (start, end) in zones.items():
            if start <= x_center <= end:
                zone_clearances[zone] = min(zone_clearances[zone], distance)
    
    return zone_clearances

def get_navigation_guidance(x_center, frame_width):
    """Get more precise directional guidance"""
    center_zone = frame_width * 0.1  # 10% tolerance for center
    frame_center = frame_width / 2
    
    if abs(x_center - frame_center) < center_zone:
        return "directly ahead"
    
    deviation = abs(x_center - frame_center) / (frame_width / 2)  # 0 to 1
    intensity = "slightly" if deviation < 0.3 else "more" if deviation < 0.6 else "far"
    direction = "left" if x_center < frame_center else "right"
    
    return f"ahead, move {intensity} to the {direction}"

def get_urgency_level(distance):
    """Determine urgency of navigation guidance"""
    if distance < 100:
        return "immediate", 1.0
    elif distance < 200:
        return "caution", 0.7
    else:
        return "notice", 0.4

class VoiceCommandHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.command_queue = queue.Queue()
        self.stop_event = Event()
        self.listen_thread = None
        
        # Initialize pygame mixer first
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self.feedback_sounds = {
                'command_recognized': pygame.mixer.Sound('assets/audio/recognized.wav'),
                'error': pygame.mixer.Sound('assets/audio/error.wav')
            }
        except Exception as e:
            print(f"Warning: Could not initialize audio feedback: {e}")
            self.feedback_sounds = None

    def _play_feedback(self, sound_type):
        if self.feedback_sounds and sound_type in self.feedback_sounds:
            try:
                self.feedback_sounds[sound_type].play()
            except:
                pass
        
    def start_listening(self):
        self.listen_thread = Thread(target=self._continuous_listen)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        
    def _continuous_listen(self):
        with sr.Microphone() as source:
            # Adjust for ambient noise
            print("Calibrating for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Voice command system ready!")
            
            while not self.stop_event.is_set():
                try:
                    # Continuous listening with shorter timeout
                    audio = self.recognizer.listen(source, phrase_time_limit=2)
                    command = self.recognizer.recognize_google(audio).lower()
                    
                    # Enhanced command recognition
                    if any(word in command for word in ["walk", "walking", "start walking"]):
                        self.command_queue.put(("mode", "walking"))
                        self._play_feedback('command_recognized')
                    elif any(word in command for word in ["normal", "stop", "stop walking"]):
                        self.command_queue.put(("mode", "normal"))
                        self._play_feedback('command_recognized')
                    elif any(phrase in command for phrase in ["what's around", "what is around", "describe", "tell me"]):
                        self.command_queue.put(("action", "describe"))
                        self._play_feedback('command_recognized')
                    if any(word in command for word in ["clear", "path", "safe distance"]):
                        self.command_queue.put(("action", "clearance"))
                    if any(word in command for word in ["navigate", "navigation", "guide me"]):
                        self.command_queue.put(("mode", "navigation"))
                    
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    continue
                except Exception as e:
                    print(f"Error in voice recognition: {e}")
                    self._play_feedback('error')
                    continue
    
    def get_command(self):
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop(self):
        self.stop_event.set()
        if self.listen_thread:
            self.listen_thread.join(timeout=1)

# Initialize pygame mixer before creating any Sound objects
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# Initialize components
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
model = YOLO('yolo11n.pt')

# Setup command handling
voice_handler = VoiceCommandHandler()
voice_handler.start_listening()

# Reference object (assumed values)
KNOWN_WIDTH = 45  # cm (average human width)
FOCAL_LENGTH = 700  # Experimentally determined (adjust if needed)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Add mode tracking
current_mode = "normal"
last_announcement_time = time.time()
ANNOUNCEMENT_COOLDOWN = 2  # seconds

# Add new initialization
motion_detector = MotionDetector()
sounds = init_audio()
previous_threats = []
environment_check_interval = 30  # frames
frame_count = 0

# Add navigation mode settings
NAVIGATION_ANNOUNCEMENT_COOLDOWN = 1.5  # More frequent updates in navigation mode
last_navigation_announcement = 0

def detect_surfaces(frame):
    """Detect walls and large surfaces using edge detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    surfaces = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            
            # Classify lines as vertical (walls) or horizontal (floors/ceilings)
            if 80 <= angle <= 100:  # Vertical lines
                surfaces.append(('wall', (x1, y1, x2, y2)))
            elif 0 <= angle <= 10 or 170 <= angle <= 180:  # Horizontal lines
                y_pos = (y1 + y2) / 2
                if y_pos < frame.shape[0] / 2:
                    surfaces.append(('ceiling', (x1, y1, x2, y2)))
                else:
                    surfaces.append(('floor', (x1, y1, x2, y2)))
    
    return surfaces

def get_improved_position(x_center, frame_width):
    """Get more accurate directional guidance"""
    # Define zones with percentages
    left_zone = frame_width * 0.4
    right_zone = frame_width * 0.6
    far_left = frame_width * 0.2
    far_right = frame_width * 0.8
    
    if x_center < far_left:
        return "far to your left"
    elif x_center < left_zone:
        return "to your left"
    elif x_center > far_right:
        return "far to your right"
    elif x_center > right_zone:
        return "to your right"
    else:
        return "directly in front of you"

def calculate_safe_path(frame, threats, surfaces):
    """Calculate the safest path direction"""
    frame_height, frame_width = frame.shape[:2]
    danger_map = np.zeros((frame_height, frame_width))
    
    # Add threat influences to danger map
    for threat in threats:
        x_center = threat['coords'][0]
        distance = threat['distance']
        influence = 1.0 / max(distance, 50)  # Avoid division by zero
        cv2.circle(danger_map, (int(x_center), frame_height-50), 
                  int(100 * influence), 1, -1)
    
    # Add surface influences to danger map
    for surface in surfaces:
        x1, y1, x2, y2 = surface['coords']
        depth = surface['depth']
        influence = 1.0 / max(depth, 50)
        cv2.rectangle(danger_map, 
                     (int(x1), int(y1)), 
                     (int(x2), int(y2)), 
                     influence, -1)
    
    # Find safest direction
    column_sums = np.sum(danger_map, axis=0)
    safest_x = np.argmin(column_sums)
    return get_improved_position(safest_x, frame_width)

def detect_surfaces_improved(frame):
    """Enhanced surface detection using multiple techniques"""
    height, width = frame.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Enhance edges using morphological operations
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    surfaces = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:  # Filter small contours
            continue
            
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w)/h if h != 0 else 0
        
        # Analyze shape and position
        if len(approx) > 4:  # Complex shape
            continue
            
        # Classify surface type
        if aspect_ratio > 2.5:  # Horizontal surface
            if y < height/2:
                surface_type = 'ceiling'
            else:
                surface_type = 'floor'
        elif 0.5 <= aspect_ratio <= 2.0:  # Vertical surface
            surface_type = 'wall'
            if w < width/4:  # Narrow vertical surface
                surface_type = 'door'
        else:
            continue
        
        # Calculate depth approximation
        depth = estimate_surface_depth(frame, (x, y, w, h))
        
        surfaces.append({
            'type': surface_type,
            'coords': (x, y, x+w, y+h),
            'depth': depth,
            'area': area
        })
    
    return surfaces

def estimate_surface_depth(frame, rect):
    """Estimate depth of surface using size and position"""
    x, y, w, h = rect
    height, width = frame.shape[:2]
    
    # Use size and position to estimate depth
    relative_size = (w * h) / (width * height)
    relative_position = y / height
    
    # Combine factors for depth estimation
    depth = int((1 / relative_size) * (1 + relative_position) * 100)
    return min(max(depth, 50), 500)  # Limit depth between 50cm and 500cm

def draw_surface_overlay(frame, surfaces):
    """Visualize detected surfaces"""
    for surface in surfaces:
        x1, y1, x2, y2 = surface['coords']
        surface_type = surface['type']
        depth = surface['depth']
        
        # Different colors for different surface types
        colors = {
            'wall': (0, 0, 255),    # Red
            'floor': (0, 255, 0),   # Green
            'ceiling': (255, 0, 0), # Blue
            'door': (0, 255, 255)   # Yellow
        }
        
        color = colors.get(surface_type, (255, 255, 255))
        
        # Draw surface boundary
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Add label with depth
        label = f"{surface_type}: {depth}cm"
        cv2.putText(frame, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_width = frame.shape[1]
    results = model(frame, verbose=False, iou=0.8)[0]
    detected_speech = []
    threats = []

    # Environment analysis
    if frame_count % environment_check_interval == 0:
        env_conditions = analyze_environment(frame)
        if env_conditions['low_light']:
            print("Low light conditions detected")
        
    # Enhanced threat detection
    detected_objects = []  # (object_name, distance, (x_center, y_center), threat_score)
    for result in results.boxes.data:
        x_min, y_min, x_max, y_max, conf, class_id = result
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        obj_name = model.names[int(class_id)]
        
        if conf < 0.2:
            continue

        obj_width = x_max - x_min
        distance = (KNOWN_WIDTH * FOCAL_LENGTH) / obj_width
        distance = round(distance)
        
        # Calculate center coordinates
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        # Get directional position as string
        position_str = get_position(x_min, x_max, frame_width)
        # Store both position string and coordinates
        position_data = {
            'direction': position_str,
            'coords': (x_center, y_center)
        }

        # Enhanced threat assessment
        if obj_name in THREAT_OBJECTS:
            threat_info = THREAT_OBJECTS[obj_name]
            
            # Check for motion if object is motion sensitive
            is_moving = False
            if threat_info['motion_sensitive']:
                is_moving = motion_detector.detect_motion(frame, (x_min, y_min, x_max, y_max))
            
            # Calculate threat score
            threat_score = threat_info['priority']
            if is_moving:
                threat_score *= 1.5
            if distance < threat_info['safe_distance']:
                threat_score *= (1 + (threat_info['safe_distance'] - distance) / threat_info['safe_distance'])
                
            # Add to threats with score
            if threat_score > 5:  # Threshold for threat consideration
                threats.append((obj_name, distance, position_data, threat_score))
                detected_objects.append((obj_name, distance, position_data['coords'], threat_score))
                # Play spatial audio for immediate threats
                if distance < threat_info['safe_distance'] * 0.5:
                    play_spatial_sound(sounds['warning'], (x_min + x_max) / 2, distance)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red for threats
        cv2.putText(frame, f"{obj_name}: {distance}cm", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        detected_speech.append(f"{obj_name} {position_str} at {distance} centimeters")

    current_time = time.time()
    command_data = voice_handler.get_command()
    if command_data:
        command_type, command_value = command_data
        if command_type == "mode":
            current_mode = command_value
            tts_engine.say(f"Switching to {command_value} mode")
            tts_engine.runAndWait()
        elif command_type == "action" and command_value == "describe":
            announcement = ", ".join(detected_speech)
            tts_engine.say(announcement)
            tts_engine.runAndWait()
        elif command_value == "clearance":
            clearances = analyze_path_clearance(frame_width, detected_objects)
            
            # Generate clearance announcement
            clear_paths = []
            for zone, distance in clearances.items():
                if distance > 500:  # More than 5 meters is considered "clear"
                    clear_paths.append(f"{zone} path clear")
                else:
                    clear_paths.append(f"{zone} path clear for {int(distance)} centimeters")
            
            announcement = ". ".join(clear_paths)
            tts_engine.say(announcement)
            tts_engine.runAndWait()

    # Handle walking mode announcements

    #In walking mode, the system will:
    # Only announce objects that are potential threats
    # Prioritize closest threats
    # Include directional information
    # Have a cooldown between announcements to prevent information overload
    # Show current mode on screen

    if current_mode == "walking" and threats and (current_time - last_announcement_time) > ANNOUNCEMENT_COOLDOWN:
        # Sort threats by score instead of just distance
        threats.sort(key=lambda x: x[3], reverse=True)
        # Use the direction string instead of coordinates for announcements
        threat_announcements = [f"{obj} {pos['direction']} at {dist} centimeters" 
                              for obj, dist, pos, score in threats[:2]]
        if threat_announcements:
            announcement = "Warning: " + ", ".join(threat_announcements)
            tts_engine.say(announcement)
            tts_engine.runAndWait()
            last_announcement_time = current_time

    # Handle navigation mode
    if current_mode == "navigation":
        # Detect surfaces
        surfaces = detect_surfaces_improved(frame)
        navigation_threats = []
        
        for result in results.boxes.data:
            x_min, y_min, x_max, y_max, conf, class_id = result
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            obj_name = model.names[int(class_id)]
            
            if conf < 0.3:  # Higher confidence threshold for navigation
                continue
            
            x_center = (x_min + x_max) / 2
            obj_width = x_max - x_min
            distance = round((KNOWN_WIDTH * FOCAL_LENGTH) / obj_width)
            
            if distance < 500:  # Only consider obstacles within 5 meters
                guidance = get_improved_position(x_center, frame_width)
                urgency, priority = get_urgency_level(distance)
                navigation_threats.append({
                    'object': obj_name,
                    'distance': distance,
                    'guidance': guidance,
                    'urgency': urgency,
                    'priority': priority,
                    'coords': (x_center, y_center)
                })
        
        # Process detected surfaces
        for surface in surfaces:
            x1, y1, x2, y2 = surface['coords']
            x_center = (x1 + x2) / 2
            distance = SURFACE_OBJECTS[surface['type']]['danger_distance']
            
            if distance < SURFACE_OBJECTS[surface['type']]['danger_distance']:
                navigation_threats.append({
                    'object': surface['type'],
                    'distance': distance,
                    'guidance': get_improved_position(x_center, frame_width),
                    'urgency': 'caution',
                    'priority': 0.8,
                    'coords': (x_center, (y1 + y2) / 2)
                })
        
        # Calculate safe path
        safe_direction = calculate_safe_path(frame, navigation_threats, surfaces)
        
        # Navigation announcements
        current_time = time.time()
        if navigation_threats and (current_time - last_navigation_announcement) > NAVIGATION_ANNOUNCEMENT_COOLDOWN:
            # Sort by priority and distance
            navigation_threats.sort(key=lambda x: (x['priority'], -x['distance']), reverse=True)
            
            # Take most important threat
            threat = navigation_threats[0]
            announcement = (f"{threat['object']} {threat['distance']} centimeters {threat['guidance']}. "
                          f"Safest path is {safe_direction}")
            
            if threat['urgency'] == "immediate":
                announcement = f"Warning! {announcement}"
            
            tts_engine.say(announcement)
            tts_engine.runAndWait()
            last_navigation_announcement = current_time
            
            # Play spatial audio based on urgency
            if threat['urgency'] == "immediate":
                play_spatial_sound(sounds['warning'], x_center, distance)
            elif threat['urgency'] == "caution":
                play_spatial_sound(sounds['proximity'], x_center, distance)
        
        # Draw navigation overlay
        cv2.line(frame, (int(frame_width/2), 0), (int(frame_width/2), frame.shape[0]), 
                (0, 255, 0), 1)  # Center line
        cv2.rectangle(frame, (int(frame_width/2 - frame_width*0.1), 0),
                     (int(frame_width/2 + frame_width*0.1), frame.shape[0]),
                     (0, 255, 0), 1)  # Safe zone

    cv2.putText(frame, f"Mode: {current_mode}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Obstacle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Visualize path clearance
    for zone, (start, end) in {'center': (frame_width/3, 2*frame_width/3),
                              'left': (0, frame_width/3),
                              'right': (2*frame_width/3, frame_width)}.items():
        # Draw path zones
        cv2.rectangle(frame, 
                     (int(start), 0),
                     (int(end), frame.shape[0]),
                     (0, 255, 0, 0.3),  # Green with transparency
                     2)

    # Update previous frame for motion detection
    motion_detector.previous_frame = frame.copy()
    frame_count += 1

# Cleanup
voice_handler.stop()
cap.release()
cv2.destroyAllWindows()
