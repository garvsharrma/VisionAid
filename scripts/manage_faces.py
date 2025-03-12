import os
import cv2
import argparse
import subprocess
import sys
import platform

def check_cmake():
    """Check if CMake is installed and accessible"""
    try:
        subprocess.run(['cmake', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_dependencies():
    """Install required dependencies with proper error handling"""
    system = platform.system()
    python_version = f"{sys.version_info.major}{sys.version_info.minor}"
    
    # Check CMake first
    if not check_cmake():
        print("\nERROR: CMake is required but not found!")
        print("\nInstallation steps:")
        if system == "Windows":
            print("1. Download CMake from https://cmake.org/download/")
            print("2. Choose the Windows x64 Installer")
            print("3. During installation, SELECT 'Add CMake to the system PATH'")
            print("4. Restart your computer after installation")
            print("\nAfter installing CMake, run this script again.")
        else:
            print("Run: sudo apt-get install cmake")
        return False

    print("\nInstalling dependencies...")
    try:
        # Install required packages first
        print("\nInstalling required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "wheel", "setuptools"])
        
        # Try multiple pre-built wheels for different Python versions
        if system == "Windows":
            dlib_wheels = [
                f"https://github.com/YuTing-Fang1999/dlib/raw/main/dlib-19.24.1-cp{python_version}-cp{python_version}-win_amd64.whl",
                "https://github.com/YuTing-Fang1999/dlib/raw/main/dlib-19.24.1-cp311-cp311-win_amd64.whl",
                "https://github.com/YuTing-Fang1999/dlib/raw/main/dlib-19.24.1-cp310-cp310-win_amd64.whl"
            ]
            
            installed = False
            for wheel in dlib_wheels:
                try:
                    print(f"\nTrying to install dlib from: {wheel}")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", wheel
                    ])
                    print("Successfully installed pre-built dlib!")
                    installed = True
                    break
                except subprocess.CalledProcessError:
                    continue
            
            if not installed:
                print("\nPre-built wheels failed. Attempting to build from source...")
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "dlib", "--no-cache-dir"
                    ])
                    print("Successfully built and installed dlib from source!")
                except subprocess.CalledProcessError as e:
                    print("\nError: Failed to install dlib")
                    print("\nPlease follow these steps:")
                    print("1. Install Visual Studio Build Tools 2019 or later:")
                    print("   https://visualstudio.microsoft.com/visual-cpp-build-tools/")
                    print("2. During installation, select 'Desktop development with C++'")
                    print("3. Restart your computer")
                    print("4. Run this script again")
                    return False
        
        # Install face_recognition after dlib
        print("\nInstalling face_recognition...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "face_recognition"])
        
        print("\nAll dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nError during installation: {e}")
        return False

def check_dependencies():
    """Check and install required dependencies"""
    try:
        import face_recognition
        return True
    except ImportError:
        print("\nFace recognition package not found.")
        print("Checking and installing requirements...")
        if install_dependencies():
            try:
                import face_recognition
                print("Face recognition is now available!")
                return True
            except ImportError:
                print("\nInstallation seemed successful but import failed.")
                print("Please try restarting your Python environment.")
                return False
        return False

def list_faces(database_path):
    """List all faces in the database"""
    if not os.path.exists(database_path):
        print("No face database found.")
        return
    
    faces = [f for f in os.listdir(database_path) if f.endswith(('.jpg', '.png'))]
    if not faces:
        print("No faces in database.")
        return
    
    print("\nStored faces:")
    for face in faces:
        name = os.path.splitext(face)[0]
        print(f"- {name}")

def remove_face(name, database_path):
    """Remove a face from the database"""
    face_path = os.path.join(database_path, f"{name}.jpg")
    if os.path.exists(face_path):
        os.remove(face_path)
        print(f"Removed {name} from database")
        return True
    print(f"Face not found: {name}")
    return False

def capture_face(name, database_path):
    """Capture a face from webcam"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not access webcam")
        return False

    print("\nPress SPACE to capture or Q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Draw guide rectangle
        height, width = frame.shape[:2]
        center_x, center_y = width//2, height//2
        rect_size = min(width, height) // 2
        top_left = (center_x - rect_size//2, center_y - rect_size//2)
        bottom_right = (center_x + rect_size//2, center_y + rect_size//2)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        
        cv2.imshow('Capture Face', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar
            try:
                import face_recognition
                face_locations = face_recognition.face_locations(frame)
                if face_locations:
                    if add_face(name, frame, database_path, from_array=True):
                        print("Face captured successfully!")
                        break
                else:
                    print("No face detected, please try again")
            except ImportError:
                print("Could not import face_recognition module")
                break

    cap.release()
    cv2.destroyAllWindows()
    return True

def add_face(name, image_data, database_path, from_array=False):
    """Add a new face to the database"""
    try:
        import face_recognition
    except ImportError:
        print("Face recognition module not available. Please install dependencies first.")
        return False

    # Load image
    if from_array:
        image = image_data
    else:
        image = face_recognition.load_image_file(image_data)

    # Verify face in image
    face_locations = face_recognition.face_locations(image)
    
    if not face_locations:
        print("No face detected in image")
        return False
    
    if len(face_locations) > 1:
        print("Multiple faces detected. Please use an image with a single face.")
        return False
    
    # Save face to database
    if not os.path.exists(database_path):
        os.makedirs(database_path)
    
    output_path = os.path.join(database_path, f"{name}.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Added {name} to face database")
    return True

def main():
    if not check_dependencies():
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Manage face recognition database")
    parser.add_argument('--add', '-a', help="Add a new face: --add 'name' 'image_path'", nargs=2)
    parser.add_argument('--capture', '-c', help="Capture face from webcam: --capture 'name'")
    parser.add_argument('--remove', '-r', help="Remove a face: --remove 'name'")
    parser.add_argument('--list', '-l', action='store_true', help="List all faces in database")
    parser.add_argument('--database', '-d', default='../assets/faces', help="Path to face database")
    
    args = parser.parse_args()
    
    if args.list:
        list_faces(args.database)
    elif args.add:
        name, image_path = args.add
        add_face(name, image_path, args.database)
    elif args.capture:
        capture_face(args.capture, args.database)
    elif args.remove:
        remove_face(args.remove, args.database)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
