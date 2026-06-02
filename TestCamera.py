import cv2
import subprocess
import numpy as np
import threading
import queue

IP       = "192.168.100.161"
USERNAME = "admin"
PASSWORD = "Moback@1202"
PORT     = 554

# Main Stream (101) = High Resolution
rtsp_url = f"rtsp://{USERNAME}:{PASSWORD}@{IP}:{PORT}/Streaming/Channels/101"

# ✅ FFmpeg command with ultra low latency settings
ffmpeg_cmd = [
    "ffmpeg",
    "-fflags", "nobuffer",           # No buffering
    "-flags", "low_delay",           # Low delay mode
    "-strict", "experimental",
    "-avioflags", "direct",          # Direct IO
    "-rtsp_transport", "tcp",        # TCP for stability
    "-i", rtsp_url,                  # Input URL
    "-vf", "scale=1920:1080",        # High Resolution 1080p
    "-f", "rawvideo",                # Raw video output
    "-pix_fmt", "bgr24",             # OpenCV format
    "-an",                           # No audio
    "-vcodec", "rawvideo",
    "-"
]

# Frame dimensions
WIDTH  = 1920
HEIGHT = 1080

# Queue to store latest frame only
frame_queue = queue.Queue(maxsize=1)

def capture_frames():
    """Capture frames from FFmpeg in separate thread"""
    process = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8
    )

    print("✅ Connected to camera!")

    while True:
        # Read raw frame bytes
        raw_frame = process.stdout.read(WIDTH * HEIGHT * 3)

        if not raw_frame:
            print("❌ Stream ended. Reconnecting...")
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**8
            )
            continue

        # Convert to numpy array
        frame = np.frombuffer(raw_frame, dtype=np.uint8)
        frame = frame.reshape((HEIGHT, WIDTH, 3))

        # Keep only latest frame
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            try:
                frame_queue.get_nowait()  # Remove old frame
            except:
                pass
            frame_queue.put(frame)

# Start capture thread
print(f"Connecting to: {IP}")
print("Press 'q' to quit\n")

thread = threading.Thread(target=capture_frames, daemon=True)
thread.start()

# Display frames in main thread
while True:
    if not frame_queue.empty():
        frame = frame_queue.get()

        # Show high resolution feed
        cv2.imshow("Hikvision HD Live Feed", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Stream closed.")