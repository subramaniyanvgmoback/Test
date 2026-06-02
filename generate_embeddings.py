import os
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "../knownfaces")
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
# Load existing embeddings
# -----------------------------
if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(NAMES_FILE):
    known_embeddings = list(np.load(EMBEDDINGS_FILE))
    known_names = list(np.load(NAMES_FILE))
    print("Loaded existing persons:", known_names)
else:
    known_embeddings = []
    known_names = []

# -----------------------------
# Process only NEW persons
# -----------------------------
for person_name in os.listdir(KNOWN_FACES_DIR):

    if person_name.capitalize() in known_names:
        print(f"Skipping {person_name} (already embedded)")
        continue

    print(f"Processing new person: {person_name}")

    person_path = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    person_embeddings = []

    for file in os.listdir(person_path):

        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            img_path = os.path.join(person_path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            face_tensors, _ = mtcnn(img_rgb, return_prob=True)

            if face_tensors is None:
                continue

            for face_tensor in face_tensors:

                face_tensor = face_tensor.unsqueeze(0).to(device)

                embedding = resnet(face_tensor).detach().cpu().numpy()[0]
                embedding = embedding / np.linalg.norm(embedding)

                person_embeddings.append(embedding)

    if len(person_embeddings) > 0:

        mean_embedding = np.mean(person_embeddings, axis=0)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

        known_embeddings.append(mean_embedding)
        known_names.append(person_name.capitalize())

        print(f"Added {person_name}")

# -----------------------------
# Save updated embeddings
# -----------------------------
np.save(EMBEDDINGS_FILE, known_embeddings)
np.save(NAMES_FILE, known_names)

print("\nUpdated persons:", known_names)
print("Embedding update complete.")