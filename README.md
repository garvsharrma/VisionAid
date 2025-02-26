# Assistive Vision Device

A real-time object detection and navigation assistance system for visually impaired people.

## Features

- Real-time object detection using YOLOv8
- Voice command interface
- Spatial audio feedback
- Multiple operation modes:
  - Normal mode (on-demand descriptions)
  - Walking mode (continuous threat detection)
- Support for both webcam and IP camera (phone camera)
- Motion detection and threat assessment
- Environmental awareness

## Prerequisites

- Python 3.8 or higher
- Webcam or IP camera (phone camera)
- Microphone
- Speakers/Headphones

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/assistive-vision-device.git
cd assistive-vision-device
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download YOLOv8 model:
```bash
# The script will automatically download the model on first run
```

5. Generate audio files:
```bash
python scripts/generate_audio.py
```

## Usage

### Using Webcam

```bash
python scripts/detect_webcam.py
```

### Using Phone Camera

1. Install IP Webcam app on your Android phone
2. Start the server in the app
3. Note the IP address shown in the app
4. Run:
```bash
python scripts/detect_image_cellcam.py
```

### Voice Commands

- "walking mode" - Enable continuous threat detection
- "normal mode" - Switch to on-demand descriptions
- "what's around" - Get description of surroundings
- "describe" - Same as "what's around"

## Navigation Mode

Navigation mode is designed to provide continuous, intelligent guidance for safe movement through the environment. It combines multiple detection systems to create a comprehensive understanding of the surroundings.

### Features

- **Precise Directional Guidance**
  - "Slightly left/right"
  - "More to the left/right"
  - "Far to the left/right"
  - "Directly ahead"

- **Surface Detection**
  - Walls
  - Doors
  - Floor obstacles
  - Ceiling hazards
  - Structural elements

- **Dynamic Path Finding**
  - Calculates safest walking path
  - Updates in real-time
  - Considers multiple obstacles
  - Identifies clear paths

- **Priority-Based Announcements**
  - Immediate threats announced first
  - Distance-based urgency levels
  - Spatial audio feedback
  - Continuous monitoring

### Usage

1. **Activate Navigation Mode**
   ```
   Voice Command: "navigate" or "guide me"
   ```

2. **Understanding Feedback**
   - Verbal guidance: "[Object] [Distance] [Direction]"
   - Warning sounds: Different tones for different threat levels
   - Spatial audio: Sound comes from direction of obstacle

3. **Distance Zones**
   - Immediate (< 1 meter): Urgent warnings
   - Caution (1-2 meters): Alert notifications
   - Notice (2-5 meters): Informational updates

4. **Path Guidance**
   - System suggests safest direction
   - Continuously updates as you move
   - Considers multiple obstacles simultaneously
   - Provides alternative routes when blocked

### Safety Features

- High-priority obstacle detection
- Real-time surface detection
- Motion tracking for moving objects
- Environmental awareness
- Multiple backup detection systems
- Continuous path recalculation

### Visual Feedback (For Assistants/Development)

- Color-coded threats
  - Red: Immediate danger
  - Yellow: Caution required
  - Green: Safe path
- Center line guide
- Safe zone indication
- Surface highlighting
- Path overlay visualization

### Example Announcements

```
"Wall 150 centimeters to your right. Safest path is slightly to the left"
"Warning! Person moving 100 centimeters directly ahead. Move far to the right"
"Door 200 centimeters ahead, move slightly left for clear path"
```

## Project Structure

```
assistive-vision-device/
├── assets/
│   └── audio/           # Generated audio files
├── scripts/
│   ├── detect_webcam.py      # Main webcam detection script
│   ├── detect_image_cellcam.py  # Phone camera detection script
│   └── generate_audio.py     # Audio generation script
├── requirements.txt
└── README.md
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV team
- PyTorch team
