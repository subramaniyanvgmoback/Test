import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "../embeddings.npy")
NAMES_FILE = os.path.join(BASE_DIR, "../names.npy")

embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=True)
names = np.load(NAMES_FILE, allow_pickle=True)

print("\nPersons in database:\n")

unique_names = sorted(set(names))
for n in unique_names:
    print("-", n)

person = input("\nEnter name to delete: ").strip()

if person not in names:
    print("Person not found.")
    exit()

# find indexes belonging to that person
indexes = [i for i, name in enumerate(names) if name == person]

print(f"\nFound {len(indexes)} embeddings for {person}")

# delete them
embeddings = np.delete(embeddings, indexes, axis=0)
names = np.delete(names, indexes)

# save back
np.save(EMBEDDINGS_FILE, embeddings)
np.save(NAMES_FILE, names)

print(f"\n{person} deleted successfully.")

# verify
names_check = np.load(NAMES_FILE, allow_pickle=True)

if person in names_check:
    print("Deletion failed.")
else:
    print("Deletion verified.")