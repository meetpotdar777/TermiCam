import cv2
import numpy as np
import sounddevice as sd

# Config
CHARS = "@#S%?*+;:,.. "
WIDTH, HEIGHT = 100, 40
ALPHA = 0.3 # Motion blur
volume_multiplier = 1.0

# This function runs in the background to listen to your mic
def audio_callback(indata, frames, time, status):
    global volume_multiplier
    volume_norm = np.linalg.norm(indata) * 10
    # Create a multiplier between 0.5 (quiet) and 2.0 (loud)
    volume_multiplier = np.clip(volume_norm, 0.5, 2.0)

# Start the audio stream
stream = sd.InputStream(callback=audio_callback)
stream.start()

cap = cv2.VideoCapture(0)
last_frame = None

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(cv2.resize(frame, (WIDTH, HEIGHT)), 1).astype(float)
    if last_frame is None: last_frame = frame
    frame = cv2.addWeighted(frame, ALPHA, last_frame, 1.0 - ALPHA, 0)
    last_frame = frame

    output = "\033[H"
    for row in frame.astype(np.uint8):
        for pixel in row:
            b, g, r = pixel
            # Apply audio volume to the brightness calculation
            brightness = int((0.299*r + 0.587*g + 0.114*b) * volume_multiplier)
            brightness = min(255, brightness) # Cap at 255
            
            char = CHARS[brightness // 22]
            color = f"\033[38;2;0;{max(50, brightness)};0m" # Matrix Green
            output += f"{color}{char}"
        output += "\n"
    
    print(output + "\033[0m", end="")
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
stream.stop()