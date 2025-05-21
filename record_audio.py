import sounddevice as sd
import soundfile as sf
import numpy as np
import os
from datetime import datetime
import time

# Audio parameters
SAMPLE_RATE = 16000
CHANNELS = 1
RECORDING_DIR = "recordings"

def ensure_recording_dir():
    """Create recordings directory if it doesn't exist"""
    if not os.path.exists(RECORDING_DIR):
        os.makedirs(RECORDING_DIR)

def get_filename():
    """Generate a filename based on current timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(RECORDING_DIR, f"recording_{timestamp}.wav")

def record_audio(duration=5):
    """Record audio from microphone and save to file"""
    # Make sure we have a place to save recordings
    ensure_recording_dir()
    
    while True:
        try:
            # Get user input
            print("\nRecording Options:")
            print("=" * 50)
            print("1. Start a new recording")
            print("2. Quit")
            print("=" * 50)
            
            choice = input("Enter your choice (1 or 2): ").strip()
            
            if choice == "2":
                print("Goodbye!")
                break
            elif choice == "1":
                # Get duration
                try:
                    duration = float(input("Enter recording duration in seconds (default 5): ") or 5)
                except ValueError:
                    print("Invalid duration. Using default 5 seconds.")
                    duration = 5
                
                print(f"\nRecording for {duration} seconds...")
                print("3...")
                time.sleep(1)
                print("2...")
                time.sleep(1)
                print("1...")
                time.sleep(1)
                print("Recording... 🔴")
                
                # Record audio
                recording = sd.rec(
                    int(duration * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS
                )
                
                # Wait for recording to complete
                sd.wait()
                
                # Generate filename and save
                filename = get_filename()
                sf.write(filename, recording, SAMPLE_RATE)
                print(f"\nSaved to: {filename}")
            else:
                print("Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\nRecording stopped.")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            continue

if __name__ == "__main__":
    record_audio() 