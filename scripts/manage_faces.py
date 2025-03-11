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
    
    # Check CMake first
    if not check_cmake():
        print("CMake is not installed. Please install CMake first:")
        if system == "Windows":
            print("1. Download CMake from https://cmake.org/download/")
            print("2. During installation, select 'Add CMake to system PATH'")
            print("3. Restart your terminal/IDE after installation")
        else:  # Linux/Unix
            print("Run: sudo apt-get install cmake")
            print("or")
            print("Run: sudo yum install cmake")
        return False

    print("Installing dependencies...")
    try:
        # Windows-specific: Try to install pre-built wheels first
        if system == "Windows":
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "--index-url=https://pypi.org/simple",
                    "--only-binary=:all:",
                    "dlib"
                ])
            except subprocess.CalledProcessError:
                print("Pre-built wheel not available, will try building from source...")
                
        # Install build dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wheel", "setuptools", "numpy"])
        
        # Install dlib
        subprocess.check_call([sys.executable, "-m", "pip", "install", "dlib"])
        
        # Install face_recognition
        subprocess.check_call([sys.executable, "-m", "pip", "install", "face_recognition"])
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nError during installation: {e}")
        print("\nManual installation steps:")
        print("1. Install CMake from cmake.org")
        print("2. Install Visual Studio Build Tools (for Windows)")
        print("3. Run: pip install dlib")
        print("4. Run: pip install face_recognition")
        return False

def check_dependencies():
    """Check and install required dependencies"""
    try:
        import face_recognition
        return True
    except ImportError:
        print("Face recognition package not found. Checking requirements...")
        return install_dependencies()

def add_face(name, image_path, database_path):
    """Add a new face to the database"""
    try:
        import face_recognition
    except ImportError:
        print("Face recognition module not available. Please install dependencies first.")
        return False

    # Load and verify face in image
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    
    if not face_locations:
        print(f"No face detected in {image_path}")
        return False
    
    if len(face_locations) > 1:
        print(f"Multiple faces detected in {image_path}. Please use an image with a single face.")
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
    parser.add_argument('--database', '-d', default='../assets/faces', help="Path to face database")
    
    args = parser.parse_args()
    
    if args.add:
        name, image_path = args.add
        add_face(name, image_path, args.database)

if __name__ == "__main__":
    main()
