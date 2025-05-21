import requests
import base64
import os
from colorama import init, Fore, Style
from pydub import AudioSegment
import io
import time

# Initialize colorama for colored terminal output
init()

def prepare_audio(file_path, max_duration=5):
    """Prepare audio file for API by converting and trimming if needed"""
    start_time = time.time()
    print(f"{Fore.BLUE}Preparing audio file...{Style.RESET_ALL}")
    
    # Load audio file
    audio = AudioSegment.from_wav(file_path)
    load_time = time.time() - start_time
    
    # Trim to max duration if needed
    if len(audio) > max_duration * 1000:  # pydub works in milliseconds
        print(f"{Fore.YELLOW}Trimming audio to {max_duration} seconds{Style.RESET_ALL}")
        audio = audio[:max_duration * 1000]
    
    # Convert to mono and reduce sample rate if needed
    if audio.channels > 1:
        audio = audio.set_channels(1)
    if audio.frame_rate > 16000:
        audio = audio.set_frame_rate(16000)
    
    # Export as WAV with reduced quality
    buffer = io.BytesIO()
    audio.export(buffer, format='wav', parameters=['-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1'])
    
    prepare_time = time.time() - start_time
    return buffer.getvalue(), {'load': load_time, 'total_prepare': prepare_time}

def test_audio_file(file_path):
    """Test emotion analysis API with a WAV file"""
    print(f"\n{Fore.CYAN}=== Testing Audio File ==={Style.RESET_ALL}")
    print(f"File: {file_path}")
    
    try:
        # Start timing
        total_start_time = time.time()
        
        # Prepare the audio file
        audio_data, prepare_times = prepare_audio(file_path)
        
        # Encode audio data in base64
        encode_start = time.time()
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        encode_time = time.time() - encode_start
        
        # Send to API
        print(f"{Fore.BLUE}Sending audio to emotion analysis API...{Style.RESET_ALL}")
        api_start = time.time()
        response = requests.post(
            'http://localhost:5000/analyze',
            data={
                'audio_data': audio_b64,
                'format': 'wav'
            }
        )
        api_time = time.time() - api_start
        
        if response.status_code == 200:
            results = response.json()
            
            # Calculate total time
            total_time = time.time() - total_start_time
            
            # Display client-side latency metrics
            print(f"\n{Fore.CYAN}=== Client-side Latency Metrics ==={Style.RESET_ALL}")
            print(f"Audio Load Time:     {prepare_times['load']*1000:.2f} ms")
            print(f"Audio Prepare Time:  {prepare_times['total_prepare']*1000:.2f} ms")
            print(f"Base64 Encode Time:  {encode_time*1000:.2f} ms")
            print(f"API Request Time:    {api_time*1000:.2f} ms")
            print(f"Total Process Time:  {total_time*1000:.2f} ms")
            print(f"{Fore.CYAN}==================={Style.RESET_ALL}\n")
            
            # Display server-side latency metrics
            if 'timing' in results:
                print(f"{Fore.CYAN}=== Server-side Latency Metrics ==={Style.RESET_ALL}")
                timing = results['timing']
                print(f"Decode Time:        {timing['decode_time_ms']:.2f} ms")
                print(f"Process Time:       {timing['process_time_ms']:.2f} ms")
                print(f"├─ Feature Extract: {timing.get('feature_time_ms', 0):.2f} ms")
                print(f"├─ Scale Features:  {timing.get('scale_time_ms', 0):.2f} ms")
                print(f"├─ Prediction:      {timing.get('predict_time_ms', 0):.2f} ms")
                print(f"└─ Result Process:  {timing.get('process_result_time_ms', 0):.2f} ms")
                print(f"Total Server Time:  {timing['total_time_ms']:.2f} ms")
                print(f"{Fore.CYAN}==================={Style.RESET_ALL}\n")
            
            # Display end-to-end metrics
            print(f"{Fore.CYAN}=== End-to-End Latency ==={Style.RESET_ALL}")
            print(f"Total Client Time:  {total_time*1000:.2f} ms")
            print(f"Total Server Time:  {results['timing']['total_time_ms']:.2f} ms")
            print(f"Network Overhead:   {(api_time*1000 - results['timing']['total_time_ms']):.2f} ms")
            print(f"{Fore.CYAN}==================={Style.RESET_ALL}\n")
            
            # Display results
            print(f"\n{Fore.GREEN}=== Emotion Analysis Results ==={Style.RESET_ALL}")
            print(f"Timestamp: {results['timestamp']}")
            print(f"Window Size: {results['window_size']}s, Stride: {results['stride']}s")
            print(f"{Fore.GREEN}================================{Style.RESET_ALL}\n")
            
            for window in results['results']:
                timestamp = window['timestamp']
                emotions = window['emotions']
                
                print(f"{Fore.YELLOW}Time: {timestamp:.1f}s{Style.RESET_ALL}")
                for emotion in emotions:
                    name = emotion['emotion']
                    conf = emotion['confidence']
                    
                    # Color code based on confidence
                    if conf > 0.7:
                        color = Fore.GREEN
                    elif conf > 0.4:
                        color = Fore.YELLOW
                    else:
                        color = Fore.RED
                        
                    # Create confidence bar
                    bar_length = int(conf * 20)
                    bar = '█' * bar_length + '░' * (20 - bar_length)
                    
                    print(f"  {name:10} {color}{bar} {conf:.1%}{Style.RESET_ALL}")
                print()
        else:
            print(f"{Fore.RED}API request failed: {response.status_code}{Style.RESET_ALL}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"{Fore.RED}Error testing audio file: {str(e)}{Style.RESET_ALL}")

if __name__ == "__main__":
    # Get file path from command line or use default
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter path to WAV file: ")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"{Fore.RED}Error: File not found: {file_path}{Style.RESET_ALL}")
        exit(1)
    
    # Test the audio file
    test_audio_file(file_path)
