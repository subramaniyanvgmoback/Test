import cv2
import os
import time
import threading
import queue
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

# -----------------------------
# Device
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "../embeddings.npy")
NAMES_FILE = os.path.join(BASE_DIR, "../names.npy")

# -----------------------------
# Models
# -----------------------------
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    min_face_size=40,
    keep_all=True,
    device=device
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# -----------------------------
# Load embeddings (normalize ONCE so dot product == cosine similarity)
# -----------------------------
known_embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
known_embeddings = known_embeddings / np.linalg.norm(known_embeddings, axis=1, keepdims=True)
known_names = np.load(NAMES_FILE).tolist()
print("Loaded persons:", known_names)

# -----------------------------
# Camera

# -----------------------------
IP       = "192.168.100.161"
USERNAME = "admin"
PASSWORD = "Moback@1202"
PORT     = 554
rtsp_url = f"rtsp://{USERNAME}:{PASSWORD}@{IP}:{PORT}/Streaming/Channels/101"

# -----------------------------
# Tunables
# -----------------------------
COSINE_THRESHOLD       = 0.70   # higher = stricter
PROCESS_EVERY_N_FRAMES = 2      # run recognition every Nth frame (higher = faster)
DETECT_DOWNSCALE       = 0.5    # detect on half-res; crops still taken from full-res
RECOGNITION_COOLDOWN   = 5

# -----------------------------
# Shared state
# -----------------------------
frame_queue  = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)
stop_event   = threading.Event()

def _put_latest(q, item):
    """Keep only the newest item in the queue."""
    if q.full():
        try: q.get_nowait()
        except queue.Empty: pass
    q.put(item)

# -----------------------------
# Thread 1 — Capture (always keeps newest frame -> kills latency buildup)
# -----------------------------
def capture_thread():
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"Connected to camera: {IP}")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed. Reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
        _put_latest(frame_queue, frame)
    cap.release()

# -----------------------------
# Thread 2 — Recognition (heavy work off the display thread)
# -----------------------------
def recognition_thread():
    frame_count = 0
    last_recognized = {}
    last_detections = []

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        frame_count += 1
        # Skip frames: reuse previous boxes so the overlay stays put
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            _put_latest(result_queue, (frame, last_detections))
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = []

        try:
            # Detect on a downscaled copy (fast), then scale boxes back to full-res
            small = cv2.resize(frame_rgb, (0, 0), fx=DETECT_DOWNSCALE, fy=DETECT_DOWNSCALE)
            boxes_small, _ = mtcnn.detect(small)

            if boxes_small is not None:
                boxes = boxes_small / DETECT_DOWNSCALE  # full-res coords

                # Extract aligned faces from the FULL-res frame (better accuracy)
                faces = mtcnn.extract(frame_rgb, boxes, save_path=None)

                if faces is not None:
                    if faces.ndim == 3:
                        faces = faces.unsqueeze(0)

                    # Batch ALL faces through ResNet in one pass
                    with torch.no_grad():
                        emb = resnet(faces.to(device)).cpu().numpy()
                    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

                    # Vectorized cosine similarity: (n_faces x n_known)
                    sims = emb @ known_embeddings.T
                    best_idx = sims.argmax(axis=1)
                    best_scores = sims[np.arange(len(best_idx)), best_idx]

                    now = time.time()
                    for i in range(len(boxes)):
                        score = float(best_scores[i])
                        if score > COSINE_THRESHOLD:
                            name = known_names[best_idx[i]]
                            if name not in last_recognized or \
                               (now - last_recognized[name]) > RECOGNITION_COOLDOWN:
                                last_recognized[name] = now
                        else:
                            name = "Unknown"
                        x1, y1, x2, y2 = map(int, boxes[i])
                        detections.append((x1, y1, x2, y2, name, score))
        except Exception as e:
            print(f"Recognition error: {e}")

        last_detections = detections
        _put_latest(result_queue, (frame, detections))

# -----------------------------
# Start threads
# -----------------------------
print(f"Connecting to: {IP}\nPress 'q' to quit\n")
threading.Thread(target=capture_thread, daemon=True).start()
threading.Thread(target=recognition_thread, daemon=True).start()

# -----------------------------
# Main thread — Display + FPS
# -----------------------------
prev_t = time.time()
fps = 0.0
shown = []

while True:
    try:
        frame, shown = result_queue.get(timeout=0.5)
    except queue.Empty:
        continue

    for (x1, y1, x2, y2, name, score) in shown:
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # FPS counter
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
    prev_t = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("AI Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_event.set()
        break

cv2.destroyAllWindows()
print("System stopped.")