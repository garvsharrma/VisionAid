import numpy as np
from scipy.io import wavfile

def generate_warning_sound():
    # Generate a sharp warning beep
    sample_rate = 44100
    duration = 0.1
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create warning tone (alternating frequencies)
    frequency1, frequency2 = 880, 660
    signal = np.sin(2 * np.pi * frequency1 * t) * 0.5
    signal += np.sin(2 * np.pi * frequency2 * t) * 0.5
    
    # Apply envelope
    envelope = np.exp(-3 * t)
    signal = signal * envelope
    
    # Normalize and convert to 16-bit integer
    signal = np.int16(signal * 32767)
    return sample_rate, signal

def generate_proximity_sound():
    # Generate a low pulsing sound
    sample_rate = 44100
    duration = 0.4
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create proximity tone (low frequency pulse)
    frequency = 220
    signal = np.sin(2 * np.pi * frequency * t)
    
    # Add pulsing effect
    pulse = np.sin(2 * np.pi * 5 * t)
    signal = signal * (0.5 + 0.5 * pulse)
    
    # Normalize and convert to 16-bit integer
    signal = np.int16(signal * 32767)
    return sample_rate, signal

def generate_direction_sound():
    # Generate a sliding pitch sound
    sample_rate = 44100
    duration = 0.3
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create direction indicator (sliding frequency)
    frequency = np.linspace(440, 880, len(t))
    signal = np.sin(2 * np.pi * frequency * t)
    
    # Apply envelope
    envelope = 1 - t/duration
    signal = signal * envelope
    
    # Normalize and convert to 16-bit integer
    signal = np.int16(signal * 32767)
    return sample_rate, signal

def generate_recognition_sound():
    # Generate a short, pleasant confirmation beep
    sample_rate = 44100
    duration = 0.15
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    frequency = np.linspace(880, 1760, len(t))
    signal = np.sin(2 * np.pi * frequency * t)
    
    envelope = 1 - t/duration
    signal = signal * envelope
    signal = np.int16(signal * 32767 * 0.3)
    return sample_rate, signal

def generate_error_sound():
    # Generate a short error sound
    sample_rate = 44100
    duration = 0.2
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    frequency = np.linspace(440, 220, len(t))
    signal = np.sin(2 * np.pi * frequency * t)
    
    envelope = 1 - t/duration
    signal = signal * envelope
    signal = np.int16(signal * 32767 * 0.3)
    return sample_rate, signal

def main():
    # Generate and save all sounds
    audio_path = "assets/audio"
    
    # Warning sound
    sample_rate, warning = generate_warning_sound()
    wavfile.write(f"{audio_path}/warning.wav", sample_rate, warning)
    
    # Proximity sound
    sample_rate, proximity = generate_proximity_sound()
    wavfile.write(f"{audio_path}/proximity.wav", sample_rate, proximity)
    
    # Direction sound
    sample_rate, direction = generate_direction_sound()
    wavfile.write(f"{audio_path}/direction.wav", sample_rate, direction)
    
    # Add new feedback sounds
    sample_rate, recognized = generate_recognition_sound()
    wavfile.write(f"{audio_path}/recognized.wav", sample_rate, recognized)
    
    sample_rate, error = generate_error_sound()
    wavfile.write(f"{audio_path}/error.wav", sample_rate, error)

if __name__ == "__main__":
    main()
    print("Audio files generated successfully!")
