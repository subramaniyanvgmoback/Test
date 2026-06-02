import cv2
import os
import threading
import torch
import numpy as np
import time
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
# Load embeddings
# -----------------------------
known_embeddings = np.load(EMBEDDINGS_FILE)
known_names = np.load(NAMES_FILE).tolist()

print("Loaded persons:", known_names)

# -----------------------------
# Camera
# -----------------------------
IP       = "192.168.100.161"
USERNAME = "admin"
PASSWORD = "Moback@1202"
PORT     = 554

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "max_delay;100000|"
    "reorder_queue_size;0"
)

rtsp_url = f"rtsp://{USERNAME}:{PASSWORD}@{IP}:{PORT}/Streaming/Channels/102"

def make_cap():
    c = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return c

print("Connecting to camera...")
cap = make_cap()
if not cap.isOpened():
    print("Failed to connect — check IP/credentials.")
    exit(1)
print("Connected. Press 'q' to quit.")

latest  = [None]
lock    = threading.Lock()
running = [True]

def _reader():
    global cap
    while running[0]:
        try:
            ret, frame = cap.read()
            if ret:
                with lock:
                    latest[0] = frame
            else:
                cap.release()
                cap = make_cap()
        except Exception:
            try:
                cap.release()
            except Exception:
                pass
            cap = make_cap()

threading.Thread(target=_reader, daemon=True).start()

COSINE_THRESHOLD = 0.70

# Prevent repeated recognition
last_recognized = {}
RECOGNITION_COOLDOWN = 5

while True:

    with lock:
        frame = latest[0]

    if frame is None:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_tensors, probs = mtcnn(frame_rgb, return_prob=True)
    boxes, _ = mtcnn.detect(frame_rgb)

    if face_tensors is not None and boxes is not None:

        for i in range(len(face_tensors)):

            face_tensor = face_tensors[i].unsqueeze(0).to(device)

            embedding = resnet(face_tensor).detach().cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)

            similarities = [
                np.dot(embedding, known_emb)
                for known_emb in known_embeddings
            ]

            best_index = np.argmax(similarities)
            best_score = similarities[best_index]

            if best_score > COSINE_THRESHOLD:
                name = known_names[best_index]
            else:
                name = "Unknown"

            x1, y1, x2, y2 = map(int, boxes[i])

            label = f"{name} {best_score:.2f}"

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,(0,255,0),2)


    cv2.imshow("AI Attendance System",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

running[0] = False
cap.release()
cv2.destroyAllWindows()