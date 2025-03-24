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
import pathlib
import pytesseract
from PIL import Image

# import face_recognition  # Add at top with other imports
global _results


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add sound settings (update cooldown)
SOUND_COOLDOWN = 3.0  # Increased from 2.0 to 3.0 seconds between sound alerts
DISTANCE_THRESHOLD_NORMAL = 150  # Reduced from 200 to 150cm for less frequent alerts
PRIORITY_THRESHOLD_NORMAL = 7  # Increased from 6 to 7 for higher threshold

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
FRAME_WIDTH = 768  # Further reduced from 416
FRAME_HEIGHT = 576  # Further reduced from 320
FRAME_RATE = 10  # Further reduced from 10
SKIP_FRAMES = 2  # Process every 3rd frame
MAX_OBJECTS = 5  # Limit number of objects to track
BATCH_SIZE = 1  # Process single frame at a time

# Add memory management thresholds
MEM_THRESHOLD = 750  # MB, optimized for 2GB RAM
FORCE_GC_THRESHOLD = 850  # MB, force cleanup at higher threshold

# Enhanced threat definitions with more context
THREAT_OBJECTS = {
    # Outdoor moving threats (high priority)
    'person': {'safe_distance': 200, 'priority': 5, 'motion_sensitive': True},
    'car': {'safe_distance': 700, 'priority': 8, 'motion_sensitive': True},
    'motorcycle': {'safe_distance': 500, 'priority': 7, 'motion_sensitive': True},
    'bicycle': {'safe_distance': 300, 'priority': 6, 'motion_sensitive': True},
    'truck': {'safe_distance': 800, 'priority': 8, 'motion_sensitive': True},
    'bus': {'safe_distance': 800, 'priority': 8, 'motion_sensitive': True},
    'dog': {'safe_distance': 200, 'priority': 6, 'motion_sensitive': True},
    
    # Outdoor static obstacles
    'pole': {'safe_distance': 150, 'priority': 4, 'motion_sensitive': False},
    'fire hydrant': {'safe_distance': 100, 'priority': 3, 'motion_sensitive': False},
    'stop sign': {'safe_distance': 150, 'priority': 4, 'motion_sensitive': False},
    'traffic light': {'safe_distance': 150, 'priority': 4, 'motion_sensitive': False},
    'bench': {'safe_distance': 150, 'priority': 3, 'motion_sensitive': False},
    'mailbox': {'safe_distance': 100, 'priority': 3, 'motion_sensitive': False},
    'tree': {'safe_distance': 150, 'priority': 4, 'motion_sensitive': False},
    'potted plant': {'safe_distance': 100, 'priority': 3, 'motion_sensitive': False},
    
    # Road/Path hazards (high priority)
    'hole': {'safe_distance': 200, 'priority': 7, 'motion_sensitive': False},
    'stairs': {'safe_distance': 200, 'priority': 6, 'motion_sensitive': False},
    'curb': {'safe_distance': 100, 'priority': 5, 'motion_sensitive': False},
    'puddle': {'safe_distance': 150, 'priority': 5, 'motion_sensitive': False},
    
    # Indoor furniture
    'chair': {'safe_distance': 100, 'priority': 3, 'motion_sensitive': False},
    'couch': {'safe_distance': 150, 'priority': 4, 'motion_sensitive': False},
    'sofa': {'safe_distance': 150, 'priority': 4, 'motion_sensitive': False},
    'bed': {'safe_distance': 150, 'priority': 4, 'motion_sensitive': False},
    'dining table': {'safe_distance': 150, 'priority': 4, 'motion_sensitive': False},
    'coffee table': {'safe_distance': 100, 'priority': 4, 'motion_sensitive': False},
    'desk': {'safe_distance': 120, 'priority': 4, 'motion_sensitive': False},
    'bookshelf': {'safe_distance': 120, 'priority': 4, 'motion_sensitive': False},
    
    # Indoor obstacles/openings
    'door': {'safe_distance': 120, 'priority': 5, 'motion_sensitive': True},  # True because doors can move
    'cabinet': {'safe_distance': 100, 'priority': 3, 'motion_sensitive': False},
    'refrigerator': {'safe_distance': 150, 'priority': 4, 'motion_sensitive': False},
    'sink': {'safe_distance': 100, 'priority': 3, 'motion_sensitive': False},
    'toilet': {'safe_distance': 100, 'priority': 3, 'motion_sensitive': False},
    'bathtub': {'safe_distance': 150, 'priority': 4, 'motion_sensitive': False},
    'window': {'safe_distance': 100, 'priority': 4, 'motion_sensitive': False},
    
    # Temporary obstacles
    'box': {'safe_distance': 100, 'priority': 3, 'motion_sensitive': False},
    'suitcase': {'safe_distance': 100, 'priority': 3, 'motion_sensitive': False},
    'bag': {'safe_distance': 80, 'priority': 2, 'motion_sensitive': False},
    'backpack': {'safe_distance': 80, 'priority': 2, 'motion_sensitive': False},
    
    # Electronics/Appliances
    'tv': {'safe_distance': 120, 'priority': 3, 'motion_sensitive': False},
    'laptop': {'safe_distance': 80, 'priority': 2, 'motion_sensitive': False},
    'microwave': {'safe_distance': 100, 'priority': 3, 'motion_sensitive': False},
    'oven': {'safe_distance': 150, 'priority': 5, 'motion_sensitive': False},
    
    # Construction/Maintenance
    'ladder': {'safe_distance': 150, 'priority': 5, 'motion_sensitive': False},
    'toolbox': {'safe_distance': 100, 'priority': 3, 'motion_sensitive': False},
    'trash can': {'safe_distance': 100, 'priority': 2, 'motion_sensitive': False},
    'construction barrier': {'safe_distance': 200, 'priority': 6, 'motion_sensitive': False},
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
    'cooldown': 2.0,  # seconds between text recognition attempts
    'min_confidence': 60,  # minimum confidence for OCR
    'min_text_size': 20,  # minimum text height in pixels
    'last_read_time': 0,
    'preprocessing': True,  # enable image preprocessing for better OCR
    'highlight_color': (0, 255, 0),  # Green for text boxes
    'font_scale': 0.6,
    'text_thickness': 2,
    'box_thickness': 2,
    'language': 'eng',  # OCR language
    'max_text_angle': 30  # maximum text rotation angle to detect
}

# Add spatial audio feedback
def init_audio():
    """Initialize audio with fallback paths and error handling"""
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.mixer.set_num_channels(8)
        
        # Get absolute path to audio files
        base_path = pathlib.Path(__file__).parent.parent / "assets" / "audio"
        
        sounds = {}
        required_sounds = {
            'warning': 'warning.wav',
            'proximity': 'proximity.wav', 
            'direction': 'direction.wav'
        }
        
        for sound_name, filename in required_sounds.items():
            sound_path = base_path / filename
            try:
                sounds[sound_name] = pygame.mixer.Sound(str(sound_path))
            except Exception as e:
                logger.warning(f"Could not load sound {sound_name}: {e}")
                # Create silent sound as fallback
                sounds[sound_name] = pygame.mixer.Sound(buffer=bytes([0]*44100))
        
        return sounds
    except Exception as e:
        logger.error(f"Failed to initialize audio: {e}")
        return None

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
    if distance < 50:
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
        
        # Initialize feedback sounds with error handling
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            
            base_path = pathlib.Path(__file__).parent.parent / "assets" / "audio"
            
            self.feedback_sounds = {}
            for sound_name in ['recognized', 'error']:
                try:
                    sound_path = base_path / f"{sound_name}.wav"
                    self.feedback_sounds[sound_name] = pygame.mixer.Sound(str(sound_path))
                except Exception as e:
                    logger.warning(f"Could not load {sound_name} sound: {e}")
                    # Create silent sound as fallback
                    self.feedback_sounds[sound_name] = pygame.mixer.Sound(buffer=bytes([0]*44100))
                    
        except Exception as e:
            logger.warning(f"Could not initialize audio feedback: {e}")
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
        MAX_RETRIES = 3
        RETRY_DELAY = 2
        connection_errors = 0
        
        with sr.Microphone() as source:
            print("Calibrating for ambient noise...")
            try:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                print("Voice command system ready!")
            except Exception as e:
                logger.error(f"Error during calibration: {e}")
                print("Failed to calibrate microphone, continuing with defaults")
            
            while not self.stop_event.is_set():
                try:
                    audio = self.recognizer.listen(source, phrase_time_limit=2)
                    command = self.recognizer.recognize_google(audio).lower()
                    connection_errors = 0  # Reset error count on success
                    
                    # Enhanced command recognition with priority to stop/idle commands
                    if any(word in command for word in ["idle", "stop", "stop navigation"]):
                        self.command_queue.put(("mode", "idle"))
                        self._play_feedback('command_recognized')
                    elif any(word in command for word in ["navigate", "navigation", "guide", "guide me", "walk", "walking"]):
                        self.command_queue.put(("mode", "navigation"))
                        self._play_feedback('command_recognized')
                    
                    # Rest of command handling
                    elif any(phrase in command for phrase in ["what's around", "what is around", "describe", "tell me"]):
                        self.command_queue.put(("action", "describe"))
                        self._play_feedback('command_recognized')
                    if any(word in command for word in ["clear", "path", "safe distance"]):
                        self.command_queue.put(("action", "clearance"))
                    if any(word in command for word in ["navigate", "navigation", "guide me"]):
                        self.command_queue.put(("mode", "navigation"))
                    if "where should I go" in command or "which way" in command:
                        self.command_queue.put(("action", "path_query"))
                    # Add face detection command
                    if any(phrase in command for phrase in ["detect faces", "recognize faces", "who is there"]):
                        self.command_queue.put(("action", "detect_faces"))
                        self._play_feedback('command_recognized')
                    # Add text reading command
                    elif "read text" in command or "read this" in command or "read screen" in command:
                        self.command_queue.put(("action", "read_text"))
                        self._play_feedback('command_recognized')
                    
                except sr.RequestError as e:
                    connection_errors += 1
                    logger.error(f"Network error in voice recognition: {e}")
                    if connection_errors >= MAX_RETRIES:
                        logger.warning("Multiple connection failures, waiting longer...")
                        time.sleep(RETRY_DELAY * 2)
                        connection_errors = 0
                    else:
                        time.sleep(RETRY_DELAY)
                except sr.UnknownValueError:
                    continue
                except ConnectionError as e:
                    logger.error(f"Connection forcibly closed: {e}")
                    time.sleep(RETRY_DELAY)
                except Exception as e:
                    logger.error(f"Error in voice recognition: {e}")
                    self._play_feedback('error')
                    time.sleep(1)  # Brief delay before retry
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
tts_engine.setProperty('rate', 250)

# Optimize model loading for Raspberry Pi
def load_optimized_model():
    try:
        if IS_RASPBERRY_PI:
            model = YOLO('yolo11n.pt', task='detect')  # Explicitly set task
            model.cpu()
            torch.set_num_threads(1)  # Limit to single thread on Pi
            # Force model to half precision
            model.model.half()
            torch.set_default_tensor_type(torch.HalfTensor)
        else:
            model = YOLO('yolo11n.pt', task='detect')
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
current_mode = "idle"  # Changed from "normal" to "idle"
last_announcement_time = time.time()
last_sound_time = time.time()  # Add this line
ANNOUNCEMENT_COOLDOWN = 2  # seconds

# Add new initialization
motion_detector = MotionDetector()
sounds = init_audio()
previous_threats = []
environment_check_interval = 30  # frames
frame_count = 0

# Enhanced Navigation Mode Settings (update thresholds)
NAVIGATION_SETTINGS = {
    'announcement_cooldown': 2.0,  # Increased from 1.2 to 2.0
    'moving_cooldown': 1.5,           # Shorter cooldown for moving threats
    'stationary_cooldown': 3.0,       # Longer cooldown for stationary objects
    'conf_threshold': 0.90,  # Increased from 0.35 to 0.75 for more confident detections
    'range': {
        'very_close': 40,    # cm
        'immediate': 100,    # cm
        'close': 150,       # Reduced from 200 to 150
        'medium': 300,      # Reduced from 350 to 300
        'far': 450         # Reduced from 500 to 450
    },
    'max_threats': 2,
    'priority_threshold': 7,  # Increased from 6 to 7
    'safe_zone_width': 0.15,  # Narrower safe zone for precise navigation
    'angle_thresholds': {
        'center': 10,      # degrees
        'slight': 20,
        'moderate': 35,
        'significant': 50
    },
    'path_width': 100,    # cm - typical path width
    'clear_path_threshold': 200,  # cm - minimum safe distance for clear path
    'block_threshold': 0.7,   # 70% of frame area must be blocked to trigger warning
    'zones': {
        'left': (0, 0.35),       # 0-35% of frame width for left zone
        'center': (0.35, 0.65),  # 35-65% of frame width for center zone
        'right': (0.65, 1.0)     # 65-100% of frame width for right zone
    },
    'zone_labels': {
        'left': 'on your left side',
        'center': 'directly ahead',
        'right': 'on your right side'
    }
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

def get_object_zone(x_center, frame_width):
    """Determine which zone an object is in based on its x position"""
    relative_pos = x_center / frame_width
    
    # Hard threshold checks to ensure no ambiguity
    if relative_pos < NAVIGATION_SETTINGS['zones']['left'][1]:
        return 'left', NAVIGATION_SETTINGS['zone_labels']['left']
    elif relative_pos > NAVIGATION_SETTINGS['zones']['right'][0]:
        return 'right', NAVIGATION_SETTINGS['zone_labels']['right']
    else:
        return 'center', NAVIGATION_SETTINGS['zone_labels']['center']

def get_enhanced_navigation_guidance(x_center, frame_width, distance, obj_width=0):
    """Improved directional guidance with clearer zones"""
    zone, position = get_object_zone(x_center, frame_width)
    
    # Distance context first
    if distance < 50:
        distance_str = "very close"
        urgency = "stop immediately"
    elif distance < 100:
        distance_str = "close"
        urgency = "slow down"
    else:
        distance_str = f"{distance} centimeters away"
        urgency = ""
    
    # Build guidance message with clearer zone indication
    message = f"Object {distance_str} {position}"
    
    # Add movement suggestion based on zone
    if urgency:
        message = f"{urgency}, {message}"
    elif zone == 'left':
        message += ", move to your right"
    elif zone == 'right':
        message += ", move to your left"
    elif zone == 'center' and distance < 200:
        message += ", step back"
    
    return message.strip()

def update_navigation_state(navigation_threats):
    """Track navigation state with smarter path blocking detection"""
    global previous_guidance, path_blocked
    
    # Get frame dimensions
    frame_width = get_frame_width()
    safe_zone_width = NAVIGATION_SETTINGS['safe_zone_width'] * frame_width
    
    # Analyze zones
    left_zone = []
    center_zone = []
    right_zone = []
    
    for threat in navigation_threats:
        x_center = threat['coords'][0]
        if x_center < frame_width/3:
            left_zone.append(threat)
        elif x_center > (2 * frame_width/3):
            right_zone.append(threat)
        else:
            center_zone.append(threat)
    
    # Check if center path is blocked
    center_blocked = any(
        t['distance'] < NAVIGATION_SETTINGS['range']['close'] 
        for t in center_zone
    )
    
    # Check if side paths are blocked
    left_blocked = all(
        t['distance'] < NAVIGATION_SETTINGS['range']['close'] 
        for t in left_zone
    ) if left_zone else False
    
    right_blocked = all(
        t['distance'] < NAVIGATION_SETTINGS['range']['close'] 
        for t in right_zone
    ) if right_zone else False
    
    # Determine if path is truly blocked
    if center_blocked:
        # Only consider path blocked if both sides are also blocked or nearly blocked
        if left_blocked and right_blocked:
            return "Warning! All paths blocked, stop immediately"
        elif not left_blocked and not right_blocked:
            # If both sides are clear, suggest the clearer side
            left_min_dist = min((t['distance'] for t in left_zone), default=float('inf'))
            right_min_dist = min((t['distance'] for t in right_zone), default=float('inf'))
            if left_min_dist > right_min_dist:
                return "Center blocked, move left"
            else:
                return "Center blocked, move right"
        else:
            # If one side is clear, use that
            if not left_blocked:
                return "Move left for clear path"
            else:
                return "Move right for clear path"
    
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
                elif any(word in command for word in ["idle", "stop"]):  # Changed "normal" to "idle"
                    self.command_queue.put(("mode", "idle"))
                    self._play_feedback('command_recognized')
                elif any(phrase in command for phrase in ["what's around", "what is around", "describe", "tell me"]):
                    self.command_queue.put(("action", "describe"))
                    self._play_feedback('command_recognized')
                elif any(word in command for word in ["clear", "path", "safe distance"]):
                    self.command_queue.put(("action", "clearance"))
                if "where should I go" in command or "which way" in command:
                    self.command_queue.put(("action", "path_query"))
                # Add face detection command
                if any(phrase in command for phrase in ["detect faces", "recognize faces", "who is there"]):
                    self.command_queue.put(("action", "detect_faces"))
                    self._play_feedback('command_recognized')
                # Add text reading command
                elif "read text" in command or "read this" in command or "read screen" in command:
                    self.command_queue.put(("action", "read_text"))
                    
            except (sr.WaitTimeoutError, sr.UnknownValueError):
                continue
            except Exception as e:
                print(f"Error in voice recognition: {e}")
                self._play_feedback('error')
                continue

VoiceCommandHandler._continuous_listen = _continuous_listen

def draw_navigation_overlay(frame, threats):
    """Draw enhanced navigation overlay with clear zone visualization"""
    width = frame.shape[1]
    height = frame.shape[0]
    
    # Draw zone boundaries with labels
    for zone, (start, end) in NAVIGATION_SETTINGS['zones'].items():
        x_start = int(start * width)
        x_end = int(end * width)
        
        # Different colors for each zone
        color = {
            'left': (0, 0, 255),   # Red for left
            'center': (0, 255, 0),  # Green for center
            'right': (255, 0, 0)    # Blue for right
        }[zone]
        
        # Draw zone boundaries
        cv2.line(frame, (x_start, 0), (x_start, height), color, 2)
        if zone != 'right':  # Draw right boundary for left and center zones
            cv2.line(frame, (x_end, 0), (x_end, height), color, 2)
            
        # Add zone labels
        zone_center = int((x_start + x_end) / 2)
        cv2.putText(frame, zone.upper(), (zone_center - 30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Draw threats
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

    def clear_all(self):
        """Clear all pending messages"""
        with self.message_queue.mutex:
            self.message_queue.queue.clear()
        self.is_speaking = False

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

# Add this function before the main loop starts
def detect_moving_threats(navigation_threats):
    """Identify and prioritize moving threats"""
    moving_threats = [
        threat for threat in navigation_threats 
        if threat['is_moving'] and threat['distance'] < NAVIGATION_SETTINGS['range']['close']
    ]
    
    if moving_threats:
        closest_moving = min(moving_threats, key=lambda x: x['distance'])
        return f"Warning! Moving {closest_moving['object']} approaching, {closest_moving['distance']} centimeters {closest_moving['guidance']}"
    return None


def analyze_path_safety(navigation_threats, frame_width, frame_height):
    """Analyze path safety and provide guidance"""
    if not navigation_threats:
        return "Path is completely clear, safe to proceed", "forward"

    # Sort threats by distance and position
    center_zone = []
    left_zone = []
    right_zone = []
    
    zone_width = frame_width / 3
    for threat in navigation_threats:
        x_center = threat['coords'][0]
        if x_center < zone_width:
            left_zone.append(threat)
        elif x_center < 2 * zone_width:
            center_zone.append(threat)
        else:
            right_zone.append(threat)

    # Check for path blocked condition
    total_threats = len(navigation_threats)
    if total_threats > 1 and all(t['distance'] < 200 for t in navigation_threats):
        blocked_area = sum(1 for t in navigation_threats if t['distance'] < 200)
        if blocked_area / total_threats > NAVIGATION_SETTINGS['block_threshold']:
            return "Path blocked in all directions, stop and wait", "blocked"

    # Check left zone threats
    if left_zone:
        closest_left = min(left_zone, key=lambda x: x['distance'])
        if closest_left['distance'] < NAVIGATION_SETTINGS['range']['close']:
            return f"{closest_left['object']} {closest_left['distance']} centimeters on your left, try moving right", "right"

    # Check right zone threats
    if right_zone:
        closest_right = min(right_zone, key=lambda x: x['distance'])
        if closest_right['distance'] < NAVIGATION_SETTINGS['range']['close']:
            return f"{closest_right['object']} {closest_right['distance']} centimeters on your right, try moving left", "left"

    # Check center zone last
    if center_zone:
        closest_center = min(center_zone, key=lambda x: x['distance'])
        if closest_center['distance'] < NAVIGATION_SETTINGS['range']['close']:
            # Check which side has more clearance
            left_dist = min((t['distance'] for t in left_zone), default=float('inf'))
            right_dist = min((t['distance'] for t in right_zone), default=float('inf'))
            if left_dist > right_dist:
                return f"{closest_center['object']} {closest_center['distance']} centimeters ahead, try moving left", "left"
            else:
                return f"{closest_center['object']} {closest_center['distance']} centimeters ahead, try moving right", "right"

    # If no immediate threats in any zone
    closest = min(navigation_threats, key=lambda x: x['distance'])
    if closest['distance'] > NAVIGATION_SETTINGS['clear_path_threshold']:
        return "Path clear, continue ahead", "forward"
    else:
        return f"{closest['object']} {closest['distance']} centimeters ahead, proceed with caution", "caution"

# Add after other global variables
MODE_SWITCH_COOLDOWN = 3.0  # seconds between mode switch announcements
last_mode_switch_time = time.time()

# Add new flag after other global variables
face_detection_requested = False

# Add after other constants
THREAT_QUEUE_CLEAR_INTERVAL = 3.0  # Clear threats every 3 seconds
last_queue_clear_time = time.time()

# Add after other constants
NO_THREAT_ANNOUNCEMENT_INTERVAL = 3.0  # Announce path clear after 3 seconds of no threats
last_threat_detected_time = time.time()

# Add text recognition settings
TEXT_RECOGNITION_SETTINGS = {
    'cooldown': 2.0,  # seconds between text recognition attempts
    'min_confidence': 50,  # minimum confidence for OCR
    'min_text_size': 20,  # minimum text height in pixels
    'last_read_time': 0,
    'preprocessing': True,  # enable image preprocessing for better OCR
    'highlight_color': (0, 255, 0),  # Green for text boxes
    'font_scale': 0.6,
    'text_thickness': 2,
    'box_thickness': 2,
    'language': 'eng',  # OCR language
    'max_text_angle': 30  # maximum text rotation angle to detect
}

def enhance_image_for_ocr(image):
    """Enhance image for better OCR results"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh)
    
    return denoised

def detect_and_read_text(frame):
    """Detect text in frame and prepare for reading"""
    try:
        # Enhance image
        if TEXT_RECOGNITION_SETTINGS['preprocessing']:
            proc_image = enhance_image_for_ocr(frame)
        else:
            proc_image = frame

        # Get detailed OCR data
        data = pytesseract.image_to_data(proc_image, output_type=pytesseract.Output.DICT)
        
        texts = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            conf = float(data['conf'][i])
            if conf > TEXT_RECOGNITION_SETTINGS['min_confidence']:
                text = data['text'][i].strip()
                if len(text) > 1:  # Filter out single characters
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    
                    if h >= TEXT_RECOGNITION_SETTINGS['min_text_size']:
                        # Draw text box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), 
                                    TEXT_RECOGNITION_SETTINGS['highlight_color'], 
                                    TEXT_RECOGNITION_SETTINGS['box_thickness'])
                        
                        # Add text above box
                        cv2.putText(frame, text, (x, y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  TEXT_RECOGNITION_SETTINGS['font_scale'],
                                  TEXT_RECOGNITION_SETTINGS['highlight_color'],
                                  TEXT_RECOGNITION_SETTINGS['text_thickness'])
                        
                        texts.append({
                            'text': text,
                            'confidence': conf,
                            'position': (x, y, w, h)
                        })
        
        # Sort text by vertical position (top to bottom)
        texts.sort(key=lambda x: x['position'][1])
        
        return texts
    
    except Exception as e:
        logger.error(f"Error in text recognition: {e}")
        return []

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
            _results = model(processed_frame, verbose=False, iou=0.4)[0]
        detected_speech = []
        threats = []

        # Environment analysis
        if frame_count % environment_check_interval == 0:
            env_conditions = analyze_environment(processed_frame)
            if env_conditions['low_light']:
                print("Low light conditions detected")
        
        # Enhanced threat detection and object description
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

            # Always add to detected_speech for description mode
            detected_speech.append(f"{obj_name} at {distance} centimeters {position_str}")
            
            # Threat assessment only for threat objects
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
                    
                    # More conservative sound alerts in idle mode
                    current_time = time.time()
                    if current_mode == "navigation":
                        if (current_time - last_sound_time >= SOUND_COOLDOWN and 
                            distance < DISTANCE_THRESHOLD_NORMAL and 
                            threat_score > PRIORITY_THRESHOLD_NORMAL and
                            is_moving):  # Only beep for moving threats in idle mode
                            audio_manager.queue_sound(sounds['warning'], (x_min + x_max) / 2, distance)
                            last_sound_time = current_time
                    else:
                        # Original behavior for other modes but with increased threshold
                        if distance < threat_info['safe_distance'] * 0.3:  # Changed from 0.5 to 0.3
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
                # Ensure mode changes are always processed, especially for idle/stop commands
                if command_value == "idle":
                    current_mode = command_value
                    tts_handler.say("Stopping navigation, switching to idle mode")
                    last_mode_switch_time = current_time
                    # Clear any pending navigation announcements
                    tts_handler.clear_all()
                # Only apply cooldown for other mode changes
                elif current_time - last_mode_switch_time > MODE_SWITCH_COOLDOWN:
                    if command_value != current_mode:
                        current_mode = command_value
                        tts_handler.say(f"Switching to {command_value} mode")
                        last_mode_switch_time = current_time
            elif command_type == "action":
                if command_value == "describe":
                    if detected_speech:
                        announcement = ", ".join(detected_speech)
                        tts_handler.say(f"I can see: {announcement}")
                    else:
                        tts_handler.say("No objects detected in view")
                elif command_value == "detect_faces":
                    face_detection_requested = True
                    tts_handler.say("Starting face detection")
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
                elif command_value == "read_text":
                    current_time = time.time()
                    if current_time - TEXT_RECOGNITION_SETTINGS['last_read_time'] > TEXT_RECOGNITION_SETTINGS['cooldown']:
                        detected_texts = detect_and_read_text(processed_frame)
                        if detected_texts:
                            # Group text by vertical position for more natural reading
                            text_groups = {}
                            for text in detected_texts:
                                location = text['location']
                                if location not in text_groups:
                                    text_groups[location] = []
                                text_groups[location].append(text['text'])

                            # Build announcement with position context
                            announcements = []
                            for location in ['top', 'middle', 'bottom']:
                                if location in text_groups:
                                    text_line = ' '.join(text_groups[location])
                                    announcements.append(f"At {location} of screen: {text_line}")

                            full_announcement = '. '.join(announcements)
                            tts_handler.say(f"Detected text: {full_announcement}")
                        else:
                            tts_handler.say("No text detected in view")
                        TEXT_RECOGNITION_SETTINGS['last_read_time'] = current_time

        # Handle navigation mode
        if current_mode == "navigation":
            current_time = time.time()
            navigation_threats = []
            
            # Check if no threats for 3 seconds
            if (current_time - last_threat_detected_time > NO_THREAT_ANNOUNCEMENT_INTERVAL and 
                current_time - last_announcement_time > NO_THREAT_ANNOUNCEMENT_INTERVAL):
                tts_handler.say("Path clear, safe to proceed")
                last_announcement_time = current_time
                
            # Clear threats periodically
            if current_time - last_queue_clear_time > THREAT_QUEUE_CLEAR_INTERVAL:
                navigation_threats = []
                tts_handler.clear_all()
                last_queue_clear_time = current_time

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
                    last_threat_detected_time = current_time  # Update last threat time
                
                if distance < NAVIGATION_SETTINGS['range']['immediate']:
                    # Force immediate announcement for very close objects
                    tts_handler.clear_all()
                    tts_handler.say(f"Warning! {obj_name} very close {guidance}")
                    audio_manager.queue_sound(sounds['warning'], x_center, distance)
                    last_announcement_time = current_time
                    break
            
            if navigation_threats:
                current_time = time.time()
                if current_time - last_announcement_time > NAVIGATION_SETTINGS['announcement_cooldown']:
                    direction = None  # Initialize direction variable
                    
                    # Check for moving threats first
                    moving_alert = detect_moving_threats(navigation_threats)
                    if moving_alert:
                        announcements = [moving_alert]
                    else:
                        # Get path guidance with direction
                        guidance, direction = analyze_path_safety(
                            navigation_threats,
                            get_frame_width(),
                            processed_frame.shape[0]
                        )
                        announcements = [guidance]
                    
                    if announcements:
                        tts_handler.say(". ".join(announcements))
                        last_announcement_time = current_time
                        
                        # Only play spatial audio if we have a valid direction
                        if direction:
                            if direction == "left":
                                audio_manager.queue_sound(sounds['direction'], 0, 100)
                            elif direction == "right":
                                audio_manager.queue_sound(sounds['direction'], get_frame_width(), 100)
                            elif direction == "blocked":
                                audio_manager.queue_sound(sounds['warning'], get_frame_width()/2, 50)

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

        # Face recognition - only process in idle mode
        if current_mode != "navigation" and face_detection_requested and FACE_RECOGNITION_AVAILABLE:
            try:
                current_time = time.time()
                found_faces = face_manager.detect_known_faces(processed_frame)
                if found_faces:
                    face_detection_requested = False  # Reset after finding faces
                    for face in found_faces:
                        # Draw face box
                        left, top, right, bottom = face['location']
                        cv2.rectangle(processed_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(processed_frame, face['name'], (left, top - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Only announce in idle mode with cooldown
                        if current_time - face_manager.last_face_announcement > FACE_SETTINGS['announcement_cooldown']:
                            position_str, _ = get_position(left, right, get_frame_width())
                            announcement = f"Recognized {face['name']} {position_str}"
                            tts_handler.say(announcement)
                            face_manager.last_face_announcement = current_time
                elif frame_count % (FACE_SETTINGS['check_interval'] * 10) == 0:
                    # Give up after checking several frames with no faces
                    face_detection_requested = False
                    tts_handler.say("No faces detected")
                        
            except Exception as e:
                if FACE_RECOGNITION_AVAILABLE:
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

# Update Navigation Settings
NAVIGATION_SETTINGS = {
    'announcement_cooldown': 2.0,
    'moving_cooldown': 1.5,           # Shorter cooldown for moving threats
    'stationary_cooldown': 3.0,       # Longer cooldown for stationary objects
    'conf_threshold': 0.90,
    'range': {
        'immediate': 50,    # cm
        'close': 200,       # Standard detection range
        'medium': 300,
        'far': 450
    },
    'zone_thresholds': {
        'center': 0.4,      # Center zone width (40% of frame)
        'left': 0.3,        # Left zone width (30% of frame)
        'right': 0.3        # Right zone width (30% of frame)
    },
    'clear_path_threshold': 200,  # cm - minimum distance for clear path
    'block_threshold': 0.8   # 70% of frame area must be blocked to trigger warning
}

# Add new variable to track last stationary announcement
last_stationary_announcement = time.time()

# Update the navigation mode section in main loop
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
            
            # Check for motion
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
        current_time = time.time()
        moving_threats = [t for t in navigation_threats if t['is_moving']]
        stationary_threats = [t for t in navigation_threats if not t['is_moving']]
        
        # Handle moving threats first (more urgent)
        if moving_threats and current_time - last_announcement_time > NAVIGATION_SETTINGS['moving_cooldown']:
            moving_alert = detect_moving_threats(navigation_threats)
            if moving_alert:
                tts_handler.say(moving_alert)
                last_announcement_time = current_time
        
        # Handle stationary threats with longer cooldown
        elif stationary_threats and current_time - last_stationary_announcement > NAVIGATION_SETTINGS['stationary_cooldown']:
            guidance, direction = analyze_path_safety(navigation_threats, get_frame_width(), processed_frame.shape[0])
            if guidance:
                tts_handler.say(guidance)
                last_stationary_announcement = current_time
                
                if direction:
                    if direction == "left":
                        audio_manager.queue_sound(sounds['direction'], 0, 100)
                    elif direction == "right":
                        audio_manager.queue_sound(sounds['direction'], get_frame_width(), 100)
                    elif direction == "blocked":
                        audio_manager.queue_sound(sounds['warning'], get_frame_width()/2, 50)


# Add voice command for path query
def _continuous_listen(self):
    # ...existing code...
    if "where should I go" in command or "which way" in command:
        self.command_queue.put(("action", "path_query"))
    # ...existing code...

# ...rest of existing code...

# Update Navigation Settings with clearer zone boundaries
NAVIGATION_SETTINGS = {
    'announcement_cooldown': 2.0,  # Increased from 1.2 to 2.0
    'moving_cooldown': 1.5,           # Shorter cooldown for moving threats
    'stationary_cooldown': 3.0,       # Longer cooldown for stationary objects
    'conf_threshold': 0.90,  # Increased from 0.35 to 0.75 for more confident detections
    'range': {
        'very_close': 40,    # cm
        'immediate': 100,    # cm
        'close': 150,       # Reduced from 200 to 150
        'medium': 300,      # Reduced from 350 to 300
        'far': 450         # Reduced from 500 to 450
    },
    'max_threats': 2,
    'priority_threshold': 7,  # Increased from 6 to 7
    'safe_zone_width': 0.15,  # Narrower safe zone for precise navigation
    'angle_thresholds': {
        'center': 10,      # degrees
        'slight': 20,
        'moderate': 35,
        'significant': 50
    },
    'path_width': 100,    # cm - typical path width
    'clear_path_threshold': 200,  # cm - minimum safe distance for clear path
    'block_threshold': 0.7,   # 70% of frame area must be blocked to trigger warning
    'zones': {
        'left': (0, 0.3),       # 0-30% of frame width
        'center': (0.3, 0.7),   # 30-70% of frame width
        'right': (0.7, 1.0)     # 70-100% of frame width
    },
    'zone_labels': {
        'left': 'on your left',
        'center': 'ahead of you',
        'right': 'on your right'
    }
}

def get_object_zone(x_center, frame_width):
    """Determine which zone an object is in based on its x position"""
    relative_pos = x_center / frame_width
    
    for zone, (start, end) in NAVIGATION_SETTINGS['zones'].items():
        if start <= relative_pos < end:
            return zone, NAVIGATION_SETTINGS['zone_labels'][zone]
    
    return 'center', NAVIGATION_SETTINGS['zone_labels']['center']  # Default fallback

def get_enhanced_navigation_guidance(x_center, frame_width, distance, obj_width=0):
    """Improved directional guidance with clearer zones"""
    zone, position = get_object_zone(x_center, frame_width)
    
    # Distance context first
    if distance < 50:
        distance_str = "very close"
        urgency = "stop immediately"
    elif distance < 100:
        distance_str = "close"
        urgency = "slow down"
    else:
        distance_str = f"{distance} centimeters"
        urgency = ""
    
    # Build guidance message
    message = f"{distance_str} {position}"
    if urgency:
        message = f"{urgency}, object {message}"
        
    # Add movement suggestion
    if zone == 'left':
        message += ", move right"
    elif zone == 'right':
        message += ", move left"
    elif zone == 'center' and distance < 200:
        message += ", step back"
        
    return message

def draw_navigation_overlay(frame, threats):
    """Draw enhanced navigation overlay with clear zone visualization"""
    width = frame.shape[1]
    height = frame.shape[0]
    
    # Draw zone boundaries
    for zone, (start, end) in NAVIGATION_SETTINGS['zones'].items():
        x_start = int(start * width)
        x_end = int(end * width)
        
        # Different colors for each zone
        if zone == 'left':
            color = (0, 0, 255)  # Red
        elif zone == 'center':
            color = (0, 255, 0)  # Green
        else:
            color = (255, 0, 0)  # Blue
            
        cv2.line(frame, (x_start, 0), (x_start, height), color, 2)
        cv2.putText(frame, zone, (x_start + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw threats
    for threat in threats:
        x = int(threat['coords'][0])
        y = height - int((threat['distance'] / NAVIGATION_SETTINGS['range']['far']) * height)
        color = (0, 0, 255) if threat['urgency'] == "immediate" else (0, 255, 255)
        cv2.circle(frame, (x, y), 5, color, -1)

# ...rest of existing code...