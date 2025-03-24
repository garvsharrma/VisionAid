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
from queue import Queue
from threading import Thread
import easyocr  # Add this import

# import face_recognition  # Add at top with other imports
global _results


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



# Add sound settings
SOUND_COOLDOWN = 2.0  # seconds between sound alerts
DISTANCE_THRESHOLD_NORMAL = 200  # cm - distance threshold for normal mode
PRIORITY_THRESHOLD_NORMAL = 6  # minimum priority score for alerts in normal mode

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

# Add text recognition settings after other settings
TEXT_RECOGNITION_SETTINGS = {
    'languages': ['en'],
    'gpu': torch.cuda.is_available(),
    'min_confidence': 0.4,  # Increased confidence threshold
    'announcement_cooldown': 2.0,
    'min_text_size': 15,    # Reduced minimum size
    'max_text_length': 100, # Maximum text length to process
    'rotation_angles': [0, -90, 90], # Check different orientations
    'block_merge_threshold': 10,  # Pixels between text blocks to merge
    'book_mode': {
        'min_text_height': 20,      # Minimum text height for book text
        'line_spacing': 10,         # Expected spacing between text lines
        'page_margin': 30,          # Margin from edges to detect page boundaries
        'reading_direction': 'top-down'  # Reading direction for book text
    }
}

# Add spatial audio feedback
def init_audio():
    pygame.mixer.init()
    pygame.mixer.set_num_channels(8)
    audio_path = "assets/audio"
    sounds = {}
    sound_files = {
        'warning': 'warning.wav',
        'proximity': 'proximity.wav',
        'direction': 'direction.wav'
    }
    
    for sound_name, file_name in sound_files.items():
        try:
            sounds[sound_name] = pygame.mixer.Sound(f"{audio_path}/{file_name}")
        except FileNotFoundError:
            print(f"Warning: Audio file {file_name} not found in {audio_path}")
            sounds[sound_name] = None
    
    return sounds

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

        self.text_recognition_active = False
        self.read_mode_active = False  # Add this line

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
            print("Calibrating for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Voice command system ready!")
            
            while not self.stop_event.is_set():
                try:
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
                    # Add text recognition commands
                    if any(phrase in command for phrase in ["read text", "read this", "what does it say"]):
                        self.command_queue.put(("action", "read_text"))
                        self._play_feedback('command_recognized')
                    if any(word in command for word in ["read", "reading", "start reading"]):
                        self.command_queue.put(("mode", "read"))
                        self._play_feedback('command_recognized')
                    # Add book reading commands
                    if any(phrase in command for phrase in ["read book", "start reading book"]):
                        self.command_queue.put(("mode", "book"))
                        self._play_feedback('command_recognized')
                    
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
            model = YOLO('yolov8n.pt')
            model.cpu()
            torch.set_num_threads(1)
            model.model.half()
            torch.set_default_tensor_type(torch.HalfTensor)
        else:
            model = YOLO('yolov8n.pt')
            if torch.cuda.is_available():
                model.to('cuda')
            else:
                model.cpu()
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
        global _results, _frame, _frame_width
        
        # Clear frame data
        if _frame is not None:
            del _frame
            _frame = None
        _frame_width = None
        
        # Clear results
        if _results is not None:
            del _results
            _results = None
            
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if IS_RASPBERRY_PI:
            try:
                os.system('sync')
            except:
                pass
    except Exception as e:
        logger.error(f"Memory cleanup failed: {str(e)}")

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
last_sound_time = time.time()  # Add this line
ANNOUNCEMENT_COOLDOWN = 2  # seconds

# Add new initialization
motion_detector = MotionDetector()
sounds = init_audio()
previous_threats = []
environment_check_interval = 30  # frames
frame_count = 0

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
                # Add text recognition commands
                if any(phrase in command for phrase in ["read text", "read this", "what does it say"]):
                    self.command_queue.put(("action", "read_text"))
                    self._play_feedback('command_recognized')
                    
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

# Add new TTS handler class after other class definitions
class TTSHandler:
    def __init__(self):
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.message_queue = Queue()
        self.is_speaking = False
        self.stop_flag = False
        self.start_worker()

    def start_worker(self):
        self.worker_thread = Thread(target=self._process_messages, daemon=True)
        self.worker_thread.start()

    def _process_messages(self):
        while not self.stop_flag:
            try:
                message = self.message_queue.get(timeout=0.1)
                if message:
                    self.is_speaking = True
                    self.tts_engine.say(message)
                    self.tts_engine.runAndWait()
                    self.is_speaking = False
            except queue.Empty:
                continue

    def say(self, message):
        # Don't queue duplicate messages
        if self.message_queue.empty() or message != self.message_queue.queue[-1]:
            self.message_queue.put(message)

    def stop(self):
        self.stop_flag = True
        self.message_queue.put(None)  # Unblock the worker thread
        self.worker_thread.join(timeout=1)

# Add audio manager class
class AudioManager:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.mixer.set_num_channels(8)
        self.sound_queue = Queue()
        self.current_sounds = set()
        self.start_worker()

    def start_worker(self):
        self.worker_thread = Thread(target=self._process_sounds, daemon=True)
        self.worker_thread.start()

    def _process_sounds(self):
        while True:
            try:
                sound, position, distance = self.sound_queue.get(timeout=0.1)
                if sound not in self.current_sounds:
                    self.current_sounds.add(sound)
                    self._play_spatial_sound(sound, position, distance)
                    self.current_sounds.remove(sound)
            except queue.Empty:
                continue

    def _play_spatial_sound(self, sound, position, distance):
        width = get_frame_width()
        if width is None:
            return
        left_vol = 1.0 - (position / width)
        right_vol = position / width
        volume = 1.0 - min(1.0, distance / 500)
        sound.set_volume(volume)
        channel = pygame.mixer.find_channel()
        if channel:
            channel.set_volume(left_vol, right_vol)
            channel.play(sound)

    def queue_sound(self, sound, position, distance):
        self.sound_queue.put((sound, position, distance))

# Replace global tts_engine initialization with TTSHandler
tts_handler = TTSHandler()
audio_manager = AudioManager()

class TextRecognitionHandler:
    def __init__(self):
        try:
            self.reader = easyocr.Reader(['en'], gpu=TEXT_RECOGNITION_SETTINGS['gpu'], 
                                       quantize=True)  # Add quantization for better performance
            self.last_announcement = 0
            self.text_cache = set()
            print("Text recognition initialized successfully")
        except Exception as e:
            print(f"Error initializing text recognition: {e}")
            self.reader = None
        self.current_reading_position = 0  # Track reading position for book mode
        self.last_read_text = set()  # Avoid re-reading same text

    def preprocess_frame(self, frame):
        """Preprocess frame for better text detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        return denoised

    def detect_text(self, frame):
        if self.reader is None:
            return []

        try:
            # Preprocess frame
            processed = self.preprocess_frame(frame)
            
            # Detect text in multiple orientations
            all_texts = []
            for angle in TEXT_RECOGNITION_SETTINGS['rotation_angles']:
                if angle != 0:
                    matrix = cv2.getRotationMatrix2D((frame.shape[1]/2, frame.shape[0]/2), angle, 1)
                    rotated = cv2.warpAffine(processed, matrix, (frame.shape[1], frame.shape[0]))
                else:
                    rotated = processed

                results = self.reader.readtext(rotated)
                
                for (bbox, text, prob) in results:
                    if prob < TEXT_RECOGNITION_SETTINGS['min_confidence']:
                        continue
                    
                    # Filter out very long or very short texts
                    if len(text) > TEXT_RECOGNITION_SETTINGS['max_text_length'] or len(text) < 2:
                        continue

                    # Calculate text size
                    (tl, tr, br, bl) = bbox
                    width = int(dist.euclidean(tl, tr))
                    height = int(dist.euclidean(tl, bl))
                    
                    if width < TEXT_RECOGNITION_SETTINGS['min_text_size'] or \
                       height < TEXT_RECOGNITION_SETTINGS['min_text_size']:
                        continue
                    
                    # Rotate coordinates back if needed
                    if angle != 0:
                        bbox = np.array(bbox)
                        bbox = np.hstack((bbox, np.ones((4,1))))
                        inv_matrix = cv2.getRotationMatrix2D((frame.shape[1]/2, frame.shape[0]/2), -angle, 1)
                        bbox = np.dot(inv_matrix, bbox.T).T

                    all_texts.append({
                        'text': text.strip(),
                        'confidence': prob,
                        'bbox': bbox,
                        'center': (int((bbox[0][0] + bbox[2][0])/2), 
                                 int((bbox[0][1] + bbox[2][1])/2))
                    })

            # Merge nearby text blocks
            merged_texts = self.merge_text_blocks(all_texts)
            return merged_texts

        except Exception as e:
            print(f"Error in text detection: {e}")
            return []

    def merge_text_blocks(self, texts):
        """Merge nearby text blocks for more natural reading"""
        if not texts:
            return []

        # Sort by vertical position
        texts.sort(key=lambda x: x['center'][1])
        
        merged = []
        current_group = [texts[0]]
        
        for text in texts[1:]:
            last = current_group[-1]
            # Check if text blocks are close enough to merge
            if (abs(text['center'][1] - last['center'][1]) < 
                TEXT_RECOGNITION_SETTINGS['block_merge_threshold']):
                current_group.append(text)
            else:
                # Merge current group and start new one
                merged.append(self.combine_text_blocks(current_group))
                current_group = [text]
        
        # Add last group
        if current_group:
            merged.append(self.combine_text_blocks(current_group))
        
        return merged

    def combine_text_blocks(self, blocks):
        """Combine multiple text blocks into one"""
        if not blocks:
            return None
        
        # Sort blocks by horizontal position
        blocks.sort(key=lambda x: x['center'][0])
        
        # Combine text and average confidence
        combined_text = " ".join(b['text'] for b in blocks)
        avg_confidence = sum(b['confidence'] for b in blocks) / len(blocks)
        
        # Use bounding box of first and last block
        bbox = np.array([
            blocks[0]['bbox'][0],  # Top-left from first block
            blocks[-1]['bbox'][1],  # Top-right from last block
            blocks[-1]['bbox'][2],  # Bottom-right from last block
            blocks[0]['bbox'][3]   # Bottom-left from first block
        ])
        
        return {
            'text': combined_text,
            'confidence': avg_confidence,
            'bbox': bbox,
            'center': (int((bbox[0][0] + bbox[2][0])/2), 
                      int((bbox[0][1] + bbox[2][1])/2))
        }

    def format_text_announcement(self, texts):
        """Format detected text for natural speech output"""
        if not texts:
            return "No text detected"
        
        # Sort texts by confidence but only announce the text
        texts.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Only include the text content, no confidence values
        lines = [text['text'] for text in texts]
        return ". ".join(lines)

    def format_book_text(self, texts):
        """Format book text for natural reading"""
        if not texts:
            return "No text detected on page"
        
        # Combine texts in reading order without confidence values
        lines = []
        current_line = []
        last_y = None
        
        for text in texts:
            current_y = text['y_position']
            
            # Check if this is a new line
            if last_y is not None and abs(current_y - last_y) > TEXT_RECOGNITION_SETTINGS['book_mode']['line_spacing']:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = []
            
            # Only add the text content
            current_line.append(text['text'])
            last_y = current_y
        
        # Add last line
        if current_line:
            lines.append(' '.join(current_line))
        
        return '. '.join(lines)

    def visualize_text(self, frame, texts):
        """Draw detected text boxes and content on frame"""
        try:
            for text_obj in texts:
                bbox = np.array(text_obj['bbox']).astype(int)
                
                # Draw green rectangle
                cv2.rectangle(frame, 
                            tuple(bbox[0]), 
                            tuple(bbox[2]), 
                            (0, 255, 0), 2)
                
                # Add white background for text
                (text_width, text_height), _ = cv2.getTextSize(
                    text_obj['text'], 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 2)
                cv2.rectangle(frame,
                            (bbox[0][0], bbox[0][1] - text_height - 4),
                            (bbox[0][0] + text_width, bbox[0][1]),
                            (255, 255, 255),
                            -1)
                
                # Draw text
                cv2.putText(frame,
                           text_obj['text'],
                           (bbox[0][0], bbox[0][1] - 4),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5,
                           (0, 0, 0),
                           2)
                
                # Show confidence
                conf_text = f"{int(text_obj['confidence']*100)}%"
                cv2.putText(frame,
                           conf_text,
                           (bbox[0][0], bbox[2][1] + 15),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.4,
                           (255, 255, 0),
                           1)
        except Exception as e:
            print(f"Error in text visualization: {e}")

    def detect_book_text(self, frame):
        """Specialized text detection for book pages"""
        if self.reader is None:
            return []
        
        try:
            processed = self.preprocess_frame(frame)
            
            # Detect text with focus on book layout
            results = self.reader.readtext(processed)
            book_texts = []
            
            for (bbox, text, prob) in results:
                if prob < TEXT_RECOGNITION_SETTINGS['min_confidence']:
                    continue
                
                # Calculate text size
                (tl, tr, br, bl) = bbox
                height = int(dist.euclidean(tl, bl))
                
                # Filter for book-like text
                if height < TEXT_RECOGNITION_SETTINGS['book_mode']['min_text_height']:
                    continue
                
                # Add to book texts with position info
                book_texts.append({
                    'text': text.strip(),
                    'confidence': prob,
                    'bbox': bbox,
                    'center': (int((tl[0] + br[0])/2), int((tl[1] + br[1])/2)),
                    'y_position': tl[1]  # Y coordinate for ordering
                })
            
            # Sort by vertical position for natural reading order
            book_texts.sort(key=lambda x: x['y_position'])
            return book_texts
            
        except Exception as e:
            print(f"Error in book text detection: {e}")
            return []

# Initialize text handler before the main loop (add this before the main loop)
try:
    text_recognition = TextRecognitionHandler()
    print("Text recognition initialized successfully")
except Exception as e:
    print(f"Failed to initialize text recognition: {e}")
    text_recognition = None

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

        # Process frame with error handling
        try:
            processed_frame = cv2.resize(new_frame, (FRAME_WIDTH, FRAME_HEIGHT))
            # Convert BGR to RGB
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Run inference with proper error handling
            if IS_RASPBERRY_PI:
                _results = model(processed_frame_rgb, verbose=False, conf=0.3, iou=0.5, max_det=MAX_OBJECTS)[0]
            else:
                # Ensure frame is in correct format for GPU
                if torch.cuda.is_available():
                    _results = model(processed_frame_rgb, verbose=False, conf=0.25, iou=0.45)[0]
                else:
                    _results = model(processed_frame_rgb, verbose=False, conf=0.25, iou=0.45)[0]
            
            set_current_frame(processed_frame)
        except Exception as e:
            logger.error(f"Error in frame processing: {e}")
            continue

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
                            audio_manager.queue_sound(sounds['warning'], (x_min + x_max) / 2, distance)
                            last_sound_time = current_time
                    else:
                        # Original behavior for other modes
                        if distance < threat_info['safe_distance'] * 0.5:
                            audio_manager.queue_sound(sounds['warning'], (x_min + x_max) / 2, distance)

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
                tts_handler.say(f"Switching to {command_value} mode")
                if command_value == "read":
                    voice_handler.read_mode_active = True
                else:
                    voice_handler.read_mode_active = False
            elif command_type == "action" and command_value == "describe":
                announcement = ", ".join(detected_speech)
                tts_handler.say(announcement)
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
                tts_handler.say(announcement)
            elif command_type == "action" and command_value == "read_text":
                # Process text recognition immediately
                if text_recognition is None:
                    tts_handler.say("Text recognition is not available")
                    continue
                
                try:
                    detected_texts = text_recognition.detect_text(processed_frame)
                    if detected_texts:
                        text_recognition.visualize_text(processed_frame, detected_texts)
                        announcement = text_recognition.format_text_announcement(detected_texts)
                        tts_handler.say(announcement)
                        # Start continuous reading mode if requested
                        voice_handler.read_mode_active = True
                        text_recognition.last_announcement = current_time
                    else:
                        tts_handler.say("No text detected")
                except Exception as e:
                    logger.error(f"Error processing text: {e}")
                    tts_handler.say("Error processing text")
                    continue

        # Handle read mode with proper error handling
        if current_mode == "read" and voice_handler.read_mode_active:
            try:
                if text_recognition is None:
                    tts_handler.say("Text recognition is not available")
                    voice_handler.read_mode_active = False
                    continue

                current_time = time.time()
                if current_time - text_recognition.last_announcement > TEXT_RECOGNITION_SETTINGS['announcement_cooldown']:
                    detected_texts = text_recognition.detect_text(processed_frame)
                    if detected_texts:
                        text_recognition.visualize_text(processed_frame, detected_texts)
                        announcement = text_recognition.format_text_announcement(detected_texts)
                        tts_handler.say(announcement)
                        text_recognition.last_announcement = current_time
            except Exception as e:
                logger.error(f"Error in read mode: {e}")
                continue

        # Handle book reading mode
        if current_mode == "book" and voice_handler.read_mode_active:
            try:
                if text_recognition is None:
                    tts_handler.say("Text recognition is not available")
                    voice_handler.read_mode_active = False
                    continue
                
                current_time = time.time()
                if current_time - text_recognition.last_announcement > TEXT_RECOGNITION_SETTINGS['announcement_cooldown']:
                    # Detect text specifically formatted for books
                    book_texts = text_recognition.detect_book_text(processed_frame)
                    if book_texts:
                        text_recognition.visualize_text(processed_frame, book_texts)
                        announcement = text_recognition.format_book_text(book_texts)
                        # Only announce if text has changed
                        if announcement not in text_recognition.last_read_text:
                            tts_handler.say(announcement)
                            text_recognition.last_read_text.add(announcement)
                            text_recognition.last_announcement = current_time
                    else:
                        if current_time - text_recognition.last_announcement > 5.0:
                            tts_handler.say("Please show a book page")
                            text_recognition.last_announcement = current_time
                            
            except Exception as e:
                logger.error(f"Error in book reading mode: {e}")
                continue

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
                        tts_handler.say(". ".join(announcements))
                        last_announcement_time = current_time
                    
                    # Remove the audio feedback section completely
                    if announcements:
                        tts_handler.say(". ".join(announcements))
                        last_announcement_time = current_time
            
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

        # Add frame protection in the main loop
        if tts_handler.is_speaking:
            # Skip heavy processing while speaking
            time.sleep(0.01)
            continue

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
                        tts_handler.say(announcement)
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
tts_handler.stop()
audio_manager.stop()
voice_handler.stop()
camera.release()
cv2.destroyAllWindows()

# Update FACE_SETTINGS
FACE_SETTINGS = {
    'database_path': 'assets/faces',
    'recognition_threshold': 0.6,     # More strict matching
    'min_face_size': 30,             # Increased minimum size
    'check_interval': 2,             # Check more frequently
    'announcement_cooldown': 3.0,     # More frequent announcements
    'tracking_timeout': 1.0,         # How long to track a face
    'min_detections': 3,             # Minimum detections before announcing
    'position_threshold': 0.2        # Position change threshold for re-announcing
}

class FaceRecognitionHandler:
    def __init__(self, database_path):
        self.database_path = database_path
        self.known_faces = {}
        self.known_encodings = {}
        self.face_locations = {}  # Track face locations over time
        self.face_timestamps = {} # Track when faces were last seen
        self.face_detection_count = {} # Count consistent detections
        self.last_announcement = 0
        self.enabled = FACE_RECOGNITION_AVAILABLE
        
        if self.enabled:
            self.load_known_faces()
        else:
            print("Face recognition disabled - module not available")

    def load_known_faces(self):
        """Load and encode known faces from database"""
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)
            return

        try:
            for face_file in os.listdir(self.database_path):
                if face_file.endswith(('.jpg', '.png')):
                    name = os.path.splitext(face_file)[0]
                    image_path = os.path.join(self.database_path, face_file)
                    face_image = face_recognition.load_image_file(image_path)
                    
                    # Get face locations first
                    face_locs = face_recognition.face_locations(face_image)
                    if not face_locs:
                        print(f"No face found in {face_file}")
                        continue
                    
                    # Use the first face found
                    encoding = face_recognition.face_encodings(face_image, [face_locs[0]])[0]
                    self.known_faces[name] = face_image
                    self.known_encodings[name] = encoding
                    print(f"Loaded face: {name}")
                    
        except Exception as e:
            print(f"Error loading face database: {e}")
            self.enabled = False

    def cleanup_old_tracks(self, current_time):
        """Remove old face tracks"""
        for name in list(self.face_timestamps.keys()):
            if current_time - self.face_timestamps[name] > FACE_SETTINGS['tracking_timeout']:
                self.face_locations.pop(name, None)
                self.face_timestamps.pop(name, None)
                self.face_detection_count.pop(name, None)

    def process_frame(self, frame, current_time):
        """Process a frame to detect and recognize faces"""
        if not self.enabled:
            return []

        try:
            # Only detect faces every check_interval frames
            face_locations = face_recognition.face_locations(frame)
            if not face_locations:
                self.cleanup_old_tracks(current_time)
                return []

            face_encodings = face_recognition.face_encodings(frame, face_locations)
            found_faces = []

            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                # Check face size
                face_height = bottom - top
                if face_height < FACE_SETTINGS['min_face_size']:
                    continue

                # Compare with known faces
                matches = face_recognition.compare_faces(
                    list(self.known_encodings.values()),
                    face_encoding,
                    tolerance=FACE_SETTINGS['recognition_threshold']
                )
                
                if True in matches:
                    name = list(self.known_faces.keys())[matches.index(True)]
                    face_center = ((left + right) // 2, (top + bottom) // 2)
                    
                    # Update tracking
                    if name in self.face_locations:
                        # Check if position has changed significantly
                        prev_center = self.face_locations[name]
                        if dist.euclidean(prev_center, face_center) < FACE_SETTINGS['position_threshold'] * frame.shape[1]:
                            self.face_detection_count[name] = self.face_detection_count.get(name, 0) + 1
                    
                    self.face_locations[name] = face_center
                    self.face_timestamps[name] = current_time
                    
                    # Only add faces that have been consistently detected
                    if self.face_detection_count.get(name, 0) >= FACE_SETTINGS['min_detections']:
                        found_faces.append({
                            'name': name,
                            'location': (left, top, right, bottom),
                            'center': face_center,
                            'confidence': 1.0 - dist.euclidean(
                                face_encoding,
                                self.known_encodings[name]
                            ) / 2
                        })

            self.cleanup_old_tracks(current_time)
            return found_faces

        except Exception as e:
            if self.enabled:
                print(f"Error in face recognition: {e}")
                self.enabled = False
            return []

# Initialize face handler (replace old face_manager initialization)
face_handler = FaceRecognitionHandler(FACE_SETTINGS['database_path'])

# ...existing code...

# Update the face recognition section in the main loop:
if FACE_RECOGNITION_AVAILABLE and frame_count % FACE_SETTINGS['check_interval'] == 0:
    current_time = time.time()
    found_faces = face_handler.process_frame(processed_frame, current_time)
    
    for face in found_faces:
        # Draw face box
        left, top, right, bottom = face['location']
        confidence = int(face['confidence'] * 100)
        cv2.rectangle(processed_frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(processed_frame, f"{face['name']} ({confidence}%)", 
                    (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Announce recognized faces
        if current_time - face_handler.last_announcement > FACE_SETTINGS['announcement_cooldown']:
            position_str, _ = get_position(left, right, get_frame_width())
            # announcement = f"Recognized {face['name']} {position_str}"
            announcement = f"{['name']} is {position_str}, {confidence}% confident"
            tts_handler.say(announcement)
            face_handler.last_announcement = current_time

# ...existing code...

# Add face position guidance settings after FACE_SETTINGS
FACE_POSITION_GUIDANCE = {
    'center_threshold': 0.1,  # 10% from center is considered "in front"
    'zones': [
        (-1.0, -0.5, "far to your left"),
        (-0.5, -0.2, "to your left"),
        (-0.2, -0.1, "slightly to your left"),
        (-0.1, 0.1, "in front of you"),
        (0.1, 0.2, "slightly to your right"),
        (0.2, 0.5, "to your right"),
        (0.5, 1.0, "far to your right")
    ]
}

def format_face_guidance(face, frame_width):
    """Format face recognition announcement with natural position guidance"""
    x_center = face['center'][0]
    # Calculate normalized position (-1 to 1, where 0 is center)
    relative_pos = (x_center - frame_width/2) / (frame_width/2)
    
    # Get position description
    position = "in front of you"  # default
    for start, end, desc in FACE_POSITION_GUIDANCE['zones']:
        if start <= relative_pos <= end:
            position = desc
            break
    
    # Format confidence as percentage without decimal
    confidence = int(face['confidence'] * 100)
    
    # Format distance if available
    distance_str = ""
    if 'distance' in face:
        dist = face['distance']
        if dist < 100:
            distance_str = "very close, "
        elif dist < 200:
            distance_str = "nearby, "
        
    return f"{face['name']} is {distance_str}{position}, {confidence}% confident"

# ...existing code...

# Replace the face recognition announcement section in the main loop
if FACE_RECOGNITION_AVAILABLE and frame_count % FACE_SETTINGS['check_interval'] == 0:
    current_time = time.time()
    found_faces = face_handler.process_frame(processed_frame, current_time)
    
    for face in found_faces:
        # Draw face box
        left, top, right, bottom = face['location']
        confidence = int(face['confidence'] * 100)
        cv2.rectangle(processed_frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(processed_frame, f"{face['name']} ({confidence}%)", 
                    (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Announce recognized faces with improved guidance
        if current_time - face_handler.last_announcement > FACE_SETTINGS['announcement_cooldown']:
            announcement = format_face_guidance(face, get_frame_width())
            tts_handler.say(announcement)
            face_handler.last_announcement = current_time

# ...existing code...