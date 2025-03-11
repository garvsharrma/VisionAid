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
import logging
import psutil
import os  # Add os import
# import face_recognition  # Add at top with other imports
global _results


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Platform-specific imports
IS_RASPBERRY_PI = False
try:
    from picamera2 import Picamera2
    from RPi import GPIO
    IS_RASPBERRY_PI = True
except ImportError:
    # Not running on Raspberry Pi, use regular webcam
    print("Running on PC - using regular webcam")
    Picamera2 = None
    GPIO = None

# Optimization settings for Raspberry Pi
FRAME_WIDTH = 320  # Further reduced from 416
FRAME_HEIGHT = 240  # Further reduced from 320
FRAME_RATE = 5  # Further reduced from 10
SKIP_FRAMES = 3  # Process every 3rd frame
MAX_OBJECTS = 5  # Limit number of objects to track
BATCH_SIZE = 1  # Process single frame at a time

# Add memory management thresholds
MEM_THRESHOLD = 750  # MB, optimized for 2GB RAM
FORCE_GC_THRESHOLD = 850  # MB, force cleanup at higher threshold

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

# Add Face Recognition Settings after other settings
FACE_SETTINGS = {
    'database_path': 'assets/faces',  # Directory to store known faces
    'recognition_threshold': 0.7,     # Lower = more strict matching
    'min_face_size': 20,             # Minimum face size in pixels
    'check_interval': 3,             # Check every N frames
    'announcement_cooldown': 5.0      # Seconds between face announcements
}

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

# Move global declarations to top level
_frame = None
_frame_width = None
_results = None  # Global results variable

def get_current_frame():
    global _frame
    return _frame

def set_current_frame(new_frame):
    global _frame, _frame_width
    _frame = new_frame
    if _frame is not None:
        _frame_width = _frame.shape[1]

def get_frame_width():
    return _frame_width

def play_spatial_sound(sound, position, distance):
    width = get_frame_width()
    if width is None:
        return
    left_vol = 1.0 - (position / width)
    right_vol = position / width
    volume = 1.0 - min(1.0, distance / 500)
    sound.set_volume(volume)
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
    """Improved depth estimation using stereo-like triangulation"""
    # Use multiple methods for robustness
    depths = []
    
    # Width-based estimation
    if width > 0:
        depth_width = (KNOWN_WIDTH * FOCAL_LENGTH) / width
        depths.append(depth_width)
    
    # Height-based estimation (assuming known height is 1.7m for person)
    if height > 0:
        KNOWN_HEIGHT = 170  # cm
        depth_height = (KNOWN_HEIGHT * FOCAL_LENGTH) / height
        depths.append(depth_height)
    
    # Area-based estimation
    if area > 0:
        KNOWN_AREA = KNOWN_WIDTH * 170  # approximate area
        depth_area = math.sqrt((KNOWN_AREA * FOCAL_LENGTH) / area)
        depths.append(depth_area)
    
    # Use median to filter outliers
    if depths:
        return round(np.median(depths))
    return 0

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
    """More precise position detection with 7 zones and structured guidance"""
    center = (x_min + x_max) / 2
    relative_pos = center / frame_width
    
    # More precise zone definitions with percentages and structured guidance
    zones = [
        (0.0, 0.15, "on your far left"),
        (0.15, 0.35, "on your left"),
        (0.35, 0.45, "slightly to your left"),
        (0.45, 0.55, "directly in front"),
        (0.55, 0.65, "slightly to your right"), 
        (0.65, 0.85, "on your right"),
        (0.85, 1.0, "on your far right")
    ]
    
    # Find position description
    description = "in front of you"
    for start, end, desc in zones:
        if start <= relative_pos <= end:
            description = desc
            break
            
    # Add guidance based on position
    if relative_pos < 0.45:
        guidance = "move right"
    elif relative_pos > 0.55:
        guidance = "move left"
    else:
        guidance = "stop"
        
    return description, relative_pos

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

# Optimize model loading for Raspberry Pi
def load_optimized_model():
    try:
        if IS_RASPBERRY_PI:
            model = YOLO('yolov8n.pt', task='detect')  # Explicitly set task
            model.cpu()
            torch.set_num_threads(1)  # Limit to single thread on Pi
            # Force model to half precision
            model.model.half()
            torch.set_default_tensor_type(torch.HalfTensor)
        else:
            model = YOLO('yolov8n.pt', task='detect')
            if torch.cuda.is_available():
                model.cuda()
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

model = load_optimized_model()

# Initialize camera based on platform
def initialize_camera():
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            if IS_RASPBERRY_PI and Picamera2:
                picam = Picamera2()
                config = picam.create_video_configuration(
                    main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
                    controls={"FrameRate": FRAME_RATE}
                )
                picam.configure(config)
                picam.start()
                return picam
            else:
                # Try different backends
                for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
                    cap = cv2.VideoCapture(0 + backend)
                    if cap.isOpened():
                        # Clear buffer by reading a few frames
                        for _ in range(5):
                            cap.read()
                        
                        # Set properties
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                        cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
                        return cap
                
                raise RuntimeError("Failed to open camera with any backend")
                
        except Exception as e:
            print(f"Camera initialization attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise RuntimeError("Failed to initialize camera after multiple attempts")

camera = initialize_camera()
torch.set_grad_enabled(False)  # Disable gradient computation

# Add memory management
def clear_memory():
    try:
        global _results
        current_frame = get_current_frame()
        if current_frame is not None:
            temp_frame = current_frame.copy()
            set_current_frame(None)
            set_current_frame(temp_frame)
        
        # Safely clear results
        if _results is not None:
            del _results
            _results = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        import gc
        gc.collect()
        
        if IS_RASPBERRY_PI:
            try:
                os.system('sync')
                # Only try to write to drop_caches if running as root
                if os.geteuid() == 0:
                    with open('/proc/sys/vm/drop_caches', 'w') as f:
                        f.write('1')
            except:
                pass
    except Exception as e:
        logger.error(f"Memory cleanup failed: {str(e)}")
        logger.debug("Current memory state:", exc_info=True)

# Memory monitoring
def print_memory_usage():
    import psutil
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f}MB")

def check_memory():
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_used = mem_info.rss / 1024 / 1024  # Convert to MB
        
        if mem_used > FORCE_GC_THRESHOLD:
            logger.warning(f"Memory usage critical: {mem_used:.1f}MB")
            clear_memory()
            time.sleep(0.5)  # Give system time to reclaim memory
            return False
            
        return mem_used < MEM_THRESHOLD
    except Exception as e:
        logger.error(f"Memory check failed: {e}")
        return True  # Continue if memory check fails

def get_memory_status():
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_used = mem_info.rss / 1024 / 1024  # MB
    total_mem = psutil.virtual_memory().total / 1024 / 1024  # MB
    return f"Memory: {mem_used:.1f}MB / {total_mem:.1f}MB ({(mem_used/total_mem)*100:.1f}%)"

# Setup command handling
voice_handler = VoiceCommandHandler()
voice_handler.start_listening()

# Reference object (assumed values)
KNOWN_WIDTH = 45  # cm (average human width)
FOCAL_LENGTH = 700  # Experimentally determined (adjust if needed)

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

# After NAVIGATION_ANNOUNCEMENT_COOLDOWN constant
SOUND_COOLDOWN = 2.5  # seconds between sound alerts in normal mode
DISTANCE_THRESHOLD_NORMAL = 150  # only play sounds for very close objects in normal mode
PRIORITY_THRESHOLD_NORMAL = 7  # only play sounds for high-priority threats in normal mode
last_sound_time = 0

# Navigation mode specific settings
NAVIGATION_ANNOUNCEMENT_COOLDOWN = 1.5  # More frequent updates
NAVIGATION_CONF_THRESHOLD = 0.3  # Higher confidence required
NAVIGATION_RANGE = 500  # Consider obstacles within 5 meters

# Walking mode specific settings
ANNOUNCEMENT_COOLDOWN = 2.0  # Less frequent updates
WALKING_CONF_THRESHOLD = 0.2  # Lower confidence threshold
MAX_THREAT_ANNOUNCEMENTS = 2  # Only announce top 2 threats

# Enhanced Navigation Mode Settings
NAVIGATION_SETTINGS = {
    'announcement_cooldown': 1.2,  # Faster updates for smoother guidance
    'conf_threshold': 0.35,  # Higher confidence for more reliable detection
    'range': {
        'immediate': 100,    # cm
        'close': 200,
        'medium': 350,
        'far': 500
    },
    'max_threats': 2,  # Reduced for clearer guidance
    'sound_cooldown': 1.0,  # Faster sound feedback
    'priority_threshold': 6,
    'safe_zone_width': 0.15,  # Narrower safe zone for precise navigation
    'angle_thresholds': {
        'center': 10,      # degrees
        'slight': 20,
        'moderate': 35,
        'significant': 50
    },
    'path_width': 100,    # cm - typical path width
    'clearance_threshold': 150  # cm - minimum safe clearance
}

def calculate_navigation_metrics(x_center, frame_width, distance, obj_width):
    """Enhanced navigation metrics with better spatial awareness"""
    frame_center = frame_width / 2
    
    # Calculate normalized position (-1 to 1, where 0 is center)
    normalized_pos = (x_center - frame_center) / (frame_width / 2)
    
    # Calculate FOV-aware angle (assuming 60° horizontal FOV)
    FOV = 60
    true_angle = normalized_pos * (FOV / 2)
    
    # Improve lateral distance calculation
    lateral_distance = abs(distance * math.sin(math.radians(true_angle)))
    forward_distance = distance * math.cos(math.radians(true_angle))
    
    # Enhanced collision risk calculation
    time_to_collision = forward_distance / 100  # Assuming walking speed ~1m/s
    lateral_risk = 1.0 - min(1.0, lateral_distance / 200)  # Higher risk if laterally close
    collision_risk = (1.0 / (1.0 + math.exp(-1 * (1.5 - time_to_collision)))) * (0.5 + 0.5 * lateral_risk)
    
    return {
        'angle': true_angle,
        'normalized_pos': normalized_pos,
        'lateral_distance': lateral_distance,
        'forward_distance': forward_distance,
        'collision_risk': collision_risk,
        'lateral_risk': lateral_risk,
        'time_to_collision': time_to_collision
    }

def get_enhanced_navigation_guidance(x_center, frame_width, distance, obj_width=0):
    """Improved directional guidance with more natural language"""
    metrics = calculate_navigation_metrics(x_center, frame_width, distance, obj_width)
    angle = metrics['angle']
    
    # Determine direction more naturally
    if abs(angle) < 5:  # Narrower center threshold
        direction = "continue straight ahead"
    else:
        # More natural turning instructions
        if abs(angle) < 15:
            intensity = "slightly"
        elif abs(angle) < 30:
            intensity = "gradually"
        elif abs(angle) < 45:
            intensity = ""  # No intensity for medium turns
        else:
            intensity = "sharply"
        
        turn_direction = "right" if angle < 0 else "left"
        direction = f"turn {intensity} {turn_direction}".strip()
    
    # Enhanced distance context
    if distance < 100:
        distance_context = "very close"
        urgency = "stop immediately"
    elif distance < 150:
        distance_context = "close"
        urgency = "slow down"
    elif distance < 300:
        distance_context = f"{distance} centimeters ahead"
        urgency = ""
    else:
        distance_context = f"about {round(distance/100)} meters ahead"
        urgency = ""
    
    # Construct clearer guidance
    guidance = f"Object {distance_context}"
    if metrics['lateral_distance'] > 50:  # Only add lateral position if significantly off-center
        side = "to your left" if angle < 0 else "to your right"
        guidance += f" {side}"
    
    if urgency:
        guidance = f"{urgency}, {guidance}"
    
    # Add action guidance
    if not urgency:  # Only add direction if not stopping
        guidance += f", {direction}"
    
    return guidance.strip()

def update_navigation_state(navigation_threats):
    """Track navigation state for smoother guidance"""
    global previous_guidance, path_blocked
    
    # Check if path is blocked
    center_threats = [t for t in navigation_threats if abs(t['coords'][0] - get_frame_width()/2) < 
                     NAVIGATION_SETTINGS['safe_zone_width'] * get_frame_width()]
    
    path_blocked = any(t['distance'] < NAVIGATION_SETTINGS['range']['close'] for t in center_threats)
    
    # Find best alternative path
    if path_blocked:
        left_threats = [t for t in navigation_threats if t['coords'][0] < get_frame_width()/2]
        right_threats = [t for t in navigation_threats if t['coords'][0] > get_frame_width()/2]
        
        left_clearance = min([t['distance'] for t in left_threats], default=float('inf'))
        right_clearance = min([t['distance'] for t in right_threats], default=float('inf'))
        
        if left_clearance > right_clearance:
            return "path blocked, try moving left"
        else:
            return "path blocked, try moving right"
            
    return None

# Fix VoiceCommandHandler by modifying the existing class instead of redefining
def _continuous_listen(self):
    with sr.Microphone() as source:
        print("Calibrating for ambient noise...")
        self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Voice command system ready!")
        
        while not self.stop_event.is_set():
            try:
                audio = self.recognizer.listen(source, phrase_time_limit=2)
                command = self.recognizer.recognize_google(audio).lower()
                
                # Simplified command recognition for navigation only
                if any(word in command for word in ["navigate", "navigation", "guide", "guide me", "walk", "walking"]):
                    self.command_queue.put(("mode", "navigation"))
                    self._play_feedback('command_recognized')
                elif any(word in command for word in ["normal", "stop"]):
                    self.command_queue.put(("mode", "normal"))
                    self._play_feedback('command_recognized')
                elif any(phrase in command for phrase in ["what's around", "what is around", "describe", "tell me"]):
                    self.command_queue.put(("action", "describe"))
                    self._play_feedback('command_recognized')
                elif any(word in command for word in ["clear", "path", "safe distance"]):
                    self.command_queue.put(("action", "clearance"))
                    
            except (sr.WaitTimeoutError, sr.UnknownValueError):
                continue
            except Exception as e:
                print(f"Error in voice recognition: {e}")
                self._play_feedback('error')
                continue

VoiceCommandHandler._continuous_listen = _continuous_listen

def draw_navigation_overlay(frame, threats):
    """Draw enhanced navigation overlay with threat visualization"""
    width = frame.shape[1]
    height = frame.shape[0]
    
    # Draw center line and safe zone
    safe_zone_width = int(width * NAVIGATION_SETTINGS['safe_zone_width'])
    center = width // 2
    
    cv2.line(frame, (center, 0), (center, height), (0, 255, 0), 1)
    cv2.rectangle(frame, 
                 (center - safe_zone_width//2, 0),
                 (center + safe_zone_width//2, height),
                 (0, 255, 0), 1)
    
    # Draw threat zones
    for threat in threats:
        x = int(threat['coords'][0])
        y = height - int((threat['distance'] / NAVIGATION_SETTINGS['range']['far']) * height)
        color = (0, 0, 255) if threat['urgency'] == "immediate" else (0, 255, 255)
        cv2.circle(frame, (x, y), 5, color, -1)

def get_frame(camera):
    """Get frame from camera with platform-specific handling"""
    try:
        if IS_RASPBERRY_PI and isinstance(camera, Picamera2):
            frame = camera.capture_array()
            return True, frame
        else:
            # For regular OpenCV camera
            camera.grab()  # Clear buffer
            return camera.read()
    except Exception as e:
        logger.error(f"Error getting frame: {e}")
        return False, None

# Update imports at the top
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    print("Face recognition not available. Face detection features will be disabled.")
    FACE_RECOGNITION_AVAILABLE = False

class FaceManager:
    def __init__(self, database_path):
        self.database_path = database_path
        self.known_faces = {}
        self.known_encodings = {}
        self.last_face_announcement = 0
        self.enabled = FACE_RECOGNITION_AVAILABLE
        if self.enabled:
            self.load_known_faces()
        else:
            print("Face recognition disabled - module not available")

    def load_known_faces(self):
        """Load known faces from database directory"""
        if not self.enabled:
            return
            
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)
            return

        try:
            for face_file in os.listdir(self.database_path):
                if face_file.endswith(('.jpg', '.png')):
                    name = os.path.splitext(face_file)[0]
                    image_path = os.path.join(self.database_path, face_file)
                    image = face_recognition.load_image_file(image_path)
                    encoding = face_recognition.face_encodings(image)[0]
                    self.known_faces[name] = image
                    self.known_encodings[name] = encoding
        except Exception as e:
            print(f"Error loading face database: {e}")
            self.enabled = False

    def detect_known_faces(self, frame):
        """Detect and identify known faces in frame"""
        if not self.enabled:
            return []
            
        try:
            face_locations = face_recognition.face_locations(frame)
            if not face_locations:
                return []
                
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            
            found_faces = []
            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(
                    list(self.known_encodings.values()),
                    face_encoding,
                    tolerance=FACE_SETTINGS['recognition_threshold']
                )
                
                if True in matches:
                    name = list(self.known_faces.keys())[matches.index(True)]
                    found_faces.append({
                        'name': name,
                        'location': (left, top, right, bottom),
                        'center': ((left + right) // 2, (top + bottom) // 2)
                    })
            
            return found_faces
            
        except Exception as e:
            if self.enabled:  # Only print error once
                print(f"Error in face detection: {e}")
                self.enabled = False
            return []

# Initialize face manager
face_manager = FaceManager(FACE_SETTINGS['database_path'])

# Main loop
while True:
    try:
        # Check memory before processing frame
        if not check_memory():
            mem_status = get_memory_status()
            logger.warning(f"Memory threshold exceeded - {mem_status}")
            clear_memory()
            time.sleep(0.2)
            continue

        ret, new_frame = get_frame(camera)
        if not ret or new_frame is None:
            logger.error("Failed to get frame, reinitializing camera...")
            camera.release()
            time.sleep(0.5)
            camera = initialize_camera()
            continue

        # Process frame
        processed_frame = cv2.resize(new_frame, (FRAME_WIDTH, FRAME_HEIGHT))
        set_current_frame(processed_frame)

        # Add memory status
        mem_status = get_memory_status()
        cv2.putText(processed_frame, mem_status, (10, processed_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Log memory usage every 30 frames
        if frame_count % 30 == 0:
            logger.info(mem_status)

        # Limit detections on Pi
        if IS_RASPBERRY_PI:
            _results = model(processed_frame, verbose=False, iou=0.5, conf=0.3, max_det=MAX_OBJECTS)[0]
        else:
            _results = model(processed_frame, verbose=False, iou=0.8)[0]
        detected_speech = []
        threats = []

        # Environment analysis
        if frame_count % environment_check_interval == 0:
            env_conditions = analyze_environment(processed_frame)
            if env_conditions['low_light']:
                print("Low light conditions detected")
        
        # Enhanced threat detection
        detected_objects = []  # (object_name, distance, (x_center, y_center), threat_score)
        for result in _results.boxes.data:
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
            position_str, relative_pos = get_position(x_min, x_max, get_frame_width())
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
                    is_moving = motion_detector.detect_motion(processed_frame, (x_min, y_min, x_max, y_max))
                
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
                    
                    # More conservative sound alerts in normal mode
                    current_time = time.time()
                    if current_mode == "normal":
                        if (current_time - last_sound_time >= SOUND_COOLDOWN and 
                            distance < DISTANCE_THRESHOLD_NORMAL and 
                            threat_score > PRIORITY_THRESHOLD_NORMAL):
                            play_spatial_sound(sounds['warning'], (x_min + x_max) / 2, distance)
                            last_sound_time = current_time
                    else:
                        # Original behavior for other modes
                        if distance < threat_info['safe_distance'] * 0.5:
                            play_spatial_sound(sounds['warning'], (x_min + x_max) / 2, distance)

            cv2.rectangle(processed_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red for threats
            cv2.putText(processed_frame, f"{obj_name}: {distance}cm", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if obj_name in THREAT_OBJECTS:
                detected_speech.append(f"{obj_name} at {distance} centimeters {position_str}")

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
                clearances = analyze_path_clearance(get_frame_width(), detected_objects)
                
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

        # Handle navigation mode
        if current_mode == "navigation":
            navigation_threats = []
            
            for result in _results.boxes.data:
                x_min, y_min, x_max, y_max, conf, class_id = result
                if conf < NAVIGATION_SETTINGS['conf_threshold']:
                    continue
                    
                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                obj_name = model.names[int(class_id)]
                x_center = (x_min + x_max) / 2
                distance = round((KNOWN_WIDTH * FOCAL_LENGTH) / (x_max - x_min))
                
                if distance < NAVIGATION_SETTINGS['range']['far']:
                    guidance = get_enhanced_navigation_guidance(x_center, get_frame_width(), distance)
                    urgency, priority = get_urgency_level(distance)
                    
                    # Check for motion if object is motion sensitive
                    is_moving = False
                    if obj_name in THREAT_OBJECTS and THREAT_OBJECTS[obj_name]['motion_sensitive']:
                        is_moving = motion_detector.detect_motion(processed_frame, (x_min, y_min, x_max, y_max))
                    
                    threat_score = THREAT_OBJECTS.get(obj_name, {'priority': 5})['priority']
                    if is_moving:
                        threat_score *= 1.5
                        
                    navigation_threats.append({
                        'object': obj_name,
                        'distance': distance,
                        'guidance': guidance,
                        'urgency': urgency,
                        'priority': threat_score,
                        'coords': (x_center, y_center),
                        'is_moving': is_moving
                    })
            
            if navigation_threats:
                # Process and announce navigation threats
                current_time = time.time()
                if current_time - last_announcement_time > NAVIGATION_SETTINGS['announcement_cooldown']:
                    # Sort by both distance and angle from center
                    navigation_threats.sort(key=lambda x: (
                        x['distance'] if abs(x['coords'][0] - get_frame_width()/2) < 
                        NAVIGATION_SETTINGS['safe_zone_width'] * get_frame_width() else float('inf'),
                        abs(x['coords'][0] - get_frame_width()/2)
                    ))
                    
                    # Get path status
                    path_status = update_navigation_state(navigation_threats)
                    
                    if path_status:
                        announcements = [path_status]
                    else:
                        # Take closest threat in path
                        threat = navigation_threats[0]
                        guidance = get_enhanced_navigation_guidance(
                            threat['coords'][0], 
                            get_frame_width(), 
                            threat['distance'],
                            threat['object_width'] if 'object_width' in threat else 0
                        )
                        
                        status = "moving " if threat['is_moving'] else ""
                        announcement = f"{status}{threat['object']} {guidance}"
                        if threat['urgency'] == "immediate":
                            announcement = f"Warning! {announcement}"
                        announcements = [announcement]
                    
                    if announcements:
                        tts_engine.say(". ".join(announcements))
                        tts_engine.runAndWait()
                        last_announcement_time = current_time
                    
                    # Play spatial audio based on urgency - Fixed: use navigation_threats instead of top_threats
                    for threat in navigation_threats[:NAVIGATION_SETTINGS['max_threats']]:
                        if threat['urgency'] == "immediate":
                            play_spatial_sound(sounds['warning'], threat['coords'][0], threat['distance'])
                        elif threat['urgency'] == "caution":
                            play_spatial_sound(sounds['proximity'], threat['coords'][0], threat['distance'])
            
            # Draw navigation visualization
            draw_navigation_overlay(processed_frame, navigation_threats)

        # Display frame and mode
        cv2.putText(processed_frame, f"Mode: {current_mode}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Obstacle Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Handle frame updates and cleanup
        motion_detector.previous_frame = processed_frame.copy()
        frame_count += 1

        # Visualize path clearance
        for zone, (start, end) in {'center': (get_frame_width()/3, 2*get_frame_width()/3),
                                  'left': (0, get_frame_width()/3),
                                  'right': (2*get_frame_width()/3, get_frame_width())}.items():
            # Draw path zones
            cv2.rectangle(processed_frame, 
                         (int(start), 0),
                         (int(end), processed_frame.shape[0]),
                         (0, 255, 0, 0.3),  # Green with transparency
                         2)

        # Update previous frame for motion detection
        motion_detector.previous_frame = processed_frame.copy()
        frame_count += 1

        # Periodic memory cleanup
        if frame_count % 100 == 0:
            clear_memory()
            print_memory_usage()

        # More frequent memory cleanup on Pi
        if IS_RASPBERRY_PI and frame_count % 30 == 0:  # Every 30 frames
            clear_memory()

        # Add small delay on Pi to prevent CPU overload
        if IS_RASPBERRY_PI:
            time.sleep(0.05)

        # More aggressive cleanup
        if frame_count % 15 == 0:
            if get_current_frame() is not None:
                temp_frame = get_current_frame().copy()
                clear_memory()
                set_current_frame(temp_frame)

        # Add small delay between frames
        time.sleep(0.01)


        # Face recognition
        if FACE_RECOGNITION_AVAILABLE and frame_count % FACE_SETTINGS['check_interval'] == 0:
            try:
                known_faces = face_manager.detect_known_faces(processed_frame)
                current_time = time.time()
                
                for face in known_faces:
                    # Draw face rectangle
                    left, top, right, bottom = face['location']
                    cv2.rectangle(processed_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(processed_frame, face['name'], (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Announce recognized faces with cooldown
                    if current_time - face_manager.last_face_announcement > FACE_SETTINGS['announcement_cooldown']:
                        x_center = face['center'][0]
                        position = get_position(left, right, get_frame_width())
                        announcement = f"Recognized {face['name']} {position}"
                        tts_engine.say(announcement)
                        tts_engine.runAndWait()
                        face_manager.last_face_announcement = current_time
            except Exception as e:
                if FACE_RECOGNITION_AVAILABLE:  # Only print once
                    print(f"Disabling face recognition due to error: {e}")
                    FACE_RECOGNITION_AVAILABLE = False

    except Exception as e:
        print(f"Error in main loop: {e}")
        clear_memory()
        time.sleep(0.1)
        continue

# Cleanup
voice_handler.stop()
camera.release()
cv2.destroyAllWindows()