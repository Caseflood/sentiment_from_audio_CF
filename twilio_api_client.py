import os
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather
from flask import Flask, request, Response
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
import base64
from colorama import init, Fore, Style
import time
import socket

# Initialize colorama for colored terminal output
init()

# Load environment variables
load_dotenv()

# Twilio credentials from environment variables
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

# API endpoint configuration
API_URL = os.getenv('API_URL', 'http://localhost:5000')

app = Flask(__name__)
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def find_available_port(start_port, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")

def download_audio(url, auth, max_retries=3, delay=1):
    """Download audio file from Twilio with retries"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, auth=auth)
            if response.status_code == 200:
                return response.content
            elif response.status_code == 404:
                if attempt < max_retries - 1:
                    print(f"Audio not ready yet, waiting {delay} seconds...")
                    time.sleep(delay)
                    continue
            else:
                raise Exception(f"Failed to download audio: {response.status_code}")
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error downloading audio, retrying...")
                time.sleep(delay)
                continue
            raise e

def analyze_emotions(audio_data):
    """Send audio to API for emotion analysis"""
    try:
        print(f"{Fore.BLUE}Sending audio to emotion analysis API...{Style.RESET_ALL}")
        # Encode audio data in base64
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Send to API
        response = requests.post(
            f"{API_URL}/analyze",
            data={
                'audio_data': audio_b64,
                'format': 'mp3'
            }
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed: {response.status_code}")
            
    except Exception as e:
        print(f"{Fore.RED}Error analyzing emotions: {str(e)}{Style.RESET_ALL}")
        return None

def display_emotions(results):
    """Display emotion analysis results in a nice format"""
    if not results or 'results' not in results:
        print(f"{Fore.RED}No emotion results to display{Style.RESET_ALL}")
        return
    
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

@app.route("/voice", methods=['GET', 'POST'])
def voice():
    """Handle incoming voice calls"""
    print(f"\n{Fore.CYAN}=== Incoming Call ==={Style.RESET_ALL}")
    print(f"Method: {request.method}")
    print(f"Headers: {dict(request.headers)}")
    print(f"Form data: {request.form}")
    print(f"Query params: {request.args}")
    print(f"{Fore.CYAN}==================={Style.RESET_ALL}\n")
    
    response = VoiceResponse()
    
    # First, ask the user to speak
    response.say("Please speak for emotion analysis. After you finish speaking, stay on the line for the results.", voice='Polly.Amy')
    
    # Record the audio
    response.record(
        action='/process_audio',
        method='POST',
        maxLength='10',
        playBeep=False,
        trim='trim-silence',
        recordingStatusCallback='/process_audio',
        recordingStatusCallbackEvent='completed'
    )
    
    return str(response)

@app.route("/process_audio", methods=['POST'])
def process_audio():
    """Process audio from Twilio and analyze emotions"""
    print(f"\n{Fore.CYAN}=== Processing Audio ==={Style.RESET_ALL}")
    print(f"Headers: {dict(request.headers)}")
    print(f"Form data: {request.form}")
    print(f"Query params: {request.args}")
    print(f"{Fore.CYAN}====================={Style.RESET_ALL}\n")
    
    try:
        # Get recording details
        recording_url = request.values.get('RecordingUrl')
        recording_status = request.values.get('RecordingStatus')
        recording_sid = request.values.get('RecordingSid')
        
        # Check if recording is ready
        if recording_status != 'completed':
            return '', 200
            
        # Get recording metadata
        recording = client.recordings(recording_sid).fetch()
        
        # Download with retries
        audio_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Recordings/{recording_sid}.mp3"
        audio_data = download_audio(audio_url, (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        
        # Process audio and return results
        results = analyze_emotions(audio_data)
        
        if results:
            # Display results in terminal
            display_emotions(results)
            
            # Format response for caller
            response = VoiceResponse()
            for window in results['results']:
                emotions = window['emotions']
                if emotions:
                    top_emotion = emotions[0]
                    emotion_text = f"{top_emotion['emotion']} with {top_emotion['confidence']:.0%} confidence"
                    response.say(f"At {window['timestamp']:.1f} seconds: {emotion_text}", voice='Polly.Amy')
            
            response.say("Thank you for using our emotion analysis service. Goodbye!", voice='Polly.Amy')
            response.hangup()
            return str(response)
        else:
            print(f"{Fore.RED}No emotions detected{Style.RESET_ALL}")
            response = VoiceResponse()
            response.say("Could not detect emotions. Please try speaking again.", voice='Polly.Amy')
            response.redirect('/voice')
            return str(response)
            
    except Exception as e:
        print(f"{Fore.RED}Error processing audio: {str(e)}{Style.RESET_ALL}")
        response = VoiceResponse()
        response.say("An error occurred. Please try again.", voice='Polly.Amy')
        response.redirect('/voice')
        return str(response)

def start_call(to_number):
    """Start a call to the specified number"""
    try:
        print(f"\n{Fore.GREEN}Starting call to {to_number}{Style.RESET_ALL}")
        call = client.calls.create(
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            url=f"{os.getenv('SERVER_URL')}/voice"
        )
        print(f"{Fore.GREEN}Call started successfully. SID: {call.sid}{Style.RESET_ALL}")
        return call.sid
    except Exception as e:
        print(f"{Fore.RED}Error starting call: {str(e)}{Style.RESET_ALL}")
        return None

if __name__ == "__main__":
    print(f"\n{Fore.GREEN}Starting Twilio API Client...{Style.RESET_ALL}")
    print(f"API URL: {API_URL}")
    print(f"Twilio Phone Number: {TWILIO_PHONE_NUMBER}")
    
    try:
        # Try to find an available port
        default_port = int(os.getenv('PORT', 5001))
        port = find_available_port(default_port)
        # if port != default_port:
            # print(f"{Fore.YELLOW}Port {default_port} is in use. Using port {port} instead.{Style.RESET_ALL}")
        
        # Start Flask server
        # print(f"{Fore.GREEN}Starting server on port {port}...{Style.RESET_ALL}")
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        # print(f"{Fore.RED}Error starting server: {str(e)}{Style.RESET_ALL}")
        # print(f"{Fore.YELLOW}Please make sure no other services are using the required ports.{Style.RESET_ALL}")
        exit(1) 