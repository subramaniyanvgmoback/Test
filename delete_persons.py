import os

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMBEDDINGS_FILE = os.path.join(BASE_DIR, "../embeddings.npy")
NAMES_FILE = os.path.join(BASE_DIR, "../names.npy")
FILES_FILE = os.path.join(BASE_DIR, "../files.npy")

files_deleted = []

# -------------------------------------------------
# Delete embeddings.npy
# -------------------------------------------------
if os.path.exists(EMBEDDINGS_FILE):
    os.remove(EMBEDDINGS_FILE)
    files_deleted.append("embeddings.npy")

# -------------------------------------------------
# Delete names.npy
# -------------------------------------------------
if os.path.exists(NAMES_FILE):
    os.remove(NAMES_FILE)
    files_deleted.append("names.npy")

# -------------------------------------------------
# Delete files.npy
# -------------------------------------------------
if os.path.exists(FILES_FILE):
    os.remove(FILES_FILE)
    files_deleted.append("files.npy")

# -------------------------------------------------
# Result
# -------------------------------------------------
if files_deleted:
    print("\nDeleted files:")
    for f in files_deleted:
        print("-", f)
else:
    print("\nNo embedding files found.")

print("\nFace database reset successfully.")
print("You can now regenerate embeddings.")